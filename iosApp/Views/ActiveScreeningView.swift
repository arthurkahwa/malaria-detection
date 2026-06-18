import AVFoundation
import SwiftData
import SwiftUI
import UIKit
@preconcurrency import Shared

/// Active screening view (spec §11 Home tab active screening). Owns the
/// per-tap capture-classify-persist sequence per spec §6: Capture tap
/// launches an `async` task that calls `CameraService.captureOneFrame()`,
/// hands the `ImageInput` to `ClassifierBridge.classify(_:)`, then
/// `PredictionStore.record(raw:)` to persist + audit. The task is
/// anchored to the view via `.task`/`Task` so recomposition mid-classify
/// cannot lose the prediction.
///
/// Lifecycle:
/// - View appear → `cameraService.start()` (requests permission on
///   first run; routes to the permission-denied fallback otherwise).
/// - `scenePhase` → `.background` → `cameraService.stop()` (spec §11
///   "active screening continues until ... app backgrounded").
/// - `authGate.state` → `.locked` → `cameraService.stop()` (auto-logout
///   ends the session per spec §11).
/// - End session tap → `cameraService.stop()` + return to idle.
///
/// **Simulator caveat:** the iOS Simulator does not produce real camera
/// frames. `CameraService.captureOneFrame()` will throw
/// `captureTimeout` on simulator. Capture must be exercised on real
/// iPhone hardware. Unit tests stay off the camera path entirely.
struct ActiveScreeningView: View {

    let recentPrediction: Prediction?

    @Environment(AuthGate.self) private var authGate
    @Environment(CameraService.self) private var cameraService
    @Environment(ClassifierBridge.self) private var classifierBridge
    @Environment(PredictionStore.self) private var predictionStore

    @Environment(\.scenePhase) private var scenePhase

    enum LocalState: Equatable {
        case idle
        case capturing
        case showing(predictionId: String, label: String, parasitizedProb: Double)
        case error(message: String)

        static func == (lhs: LocalState, rhs: LocalState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.capturing, .capturing):
                return true
            case (.showing(let a, _, _), .showing(let b, _, _)):
                return a == b
            case (.error(let a), .error(let b)):
                return a == b
            default:
                return false
            }
        }
    }

    @State private var localState: LocalState = .idle
    @State private var lastPersisted: Prediction? = nil
    @State private var showOverride: Bool = false
    @State private var didStart: Bool = false
    @State private var permissionDenied: Bool = false

    private var activeModelName: String {
        // Phase 11 Settings → Models adds the live picker. Until then,
        // the bundled BNLeaky Keras model is the only choice — match
        // the affordance label spec §11 mocks describe.
        "BN + LeakyReLU"
    }

    var body: some View {
        Group {
            if permissionDenied {
                permissionDeniedView
            } else {
                content
            }
        }
        .task {
            await startIfNeeded()
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background || newPhase == .inactive {
                cameraService.stop()
            }
        }
        .onChange(of: authStateKey) { _, _ in
            if case .locked = authGate.state {
                cameraService.stop()
                localState = .idle
            }
        }
        .sheet(isPresented: $showOverride) {
            if let lastPersisted {
                LiveOverrideSheet(prediction: lastPersisted)
            }
        }
    }

    // Computed key for `onChange` — `AuthGate.State` is not Equatable
    // by default, so collapse to a small enum-like int.
    private var authStateKey: Int {
        switch authGate.state {
        case .locked: return 0
        case .unlocked: return 1
        case .provisionedUnclaimed: return 2
        }
    }

    // MARK: - Layout

    private var content: some View {
        VStack(spacing: 0) {
            modelBadge
                .padding(.top, 12)
                .padding(.leading, 16)
                .padding(.trailing, 16)
                .padding(.bottom, 8)

            preview
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .padding(.leading, 16)
                .padding(.trailing, 16)

            controls
                .padding(.top, 16)
                .padding(.bottom, 24)
                .padding(.leading, 16)
                .padding(.trailing, 16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var modelBadge: some View {
        HStack(spacing: 6) {
            Image(systemName: "star.fill")
                .font(.caption2)
                .foregroundStyle(.yellow)
            Text(activeModelName)
                .font(.subheadline.weight(.medium))
            Spacer()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var preview: some View {
        Group {
            switch cameraService.state {
            case .running, .starting:
                CameraPreviewView(session: cameraService.session)
            case .failed(let message):
                VStack(spacing: 12) {
                    Image(systemName: "video.slash")
                        .font(.largeTitle)
                        .foregroundStyle(.white.opacity(0.7))
                    Text(message)
                        .font(.callout)
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                        .padding(.leading, 24)
                        .padding(.trailing, 24)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .idle, .stopped:
                VStack(spacing: 8) {
                    Image(systemName: "viewfinder")
                        .font(.largeTitle)
                        .foregroundStyle(.white.opacity(0.5))
                    Text("Camera idle")
                        .font(.callout)
                        .foregroundStyle(.white.opacity(0.7))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private var controls: some View {
        VStack(spacing: 12) {
            if case .showing(_, let label, let prob) = localState {
                predictionOverlay(label: label, parasitizedProb: prob)
            }
            if case .error(let message) = localState {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 12) {
                if case .showing = localState {
                    Button {
                        showOverride = true
                    } label: {
                        Text("Override")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .accessibilityIdentifier("activeScreening_override")
                }

                Button {
                    Task { await captureTapped() }
                } label: {
                    Text(captureButtonLabel)
                        .font(.title3.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(captureDisabled)
                .accessibilityIdentifier("activeScreening_capture")
            }

            Button {
                endSession()
            } label: {
                Text("End session")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.borderless)
            .accessibilityIdentifier("activeScreening_endSession")
        }
    }

    private var captureButtonLabel: String {
        switch localState {
        case .capturing: return "Capturing…"
        default: return "Capture"
        }
    }

    private var captureDisabled: Bool {
        if case .capturing = localState { return true }
        if cameraService.state != .running { return true }
        return false
    }

    private func predictionOverlay(label: String, parasitizedProb: Double) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(label)
                        .font(.title3.weight(.semibold))
                    Text("\(Int((parasitizedProb * 100).rounded()))%")
                        .font(.title3.monospacedDigit())
                }
                RiskBandIndicator(parasitizedProb: parasitizedProb)
            }
            Spacer()
        }
        .padding(.vertical, 10)
        .padding(.leading, 14)
        .padding(.trailing, 14)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Permission-denied fallback

    private var permissionDeniedView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Camera permission required")
                .font(.title3.weight(.semibold))
            Text("Open Settings → Malaria Detector → Camera to grant access.")
                .foregroundStyle(.secondary)
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            Spacer()
        }
        .padding(.leading, 24)
        .padding(.trailing, 24)
        .padding(.top, 24)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Capture flow

    private func startIfNeeded() async {
        guard !didStart else { return }
        didStart = true
        do {
            try await cameraService.start()
        } catch CameraService.CameraError.permissionDenied {
            permissionDenied = true
        } catch {
            localState = .error(message: error.localizedDescription)
        }
    }

    private func captureTapped() async {
        guard case .running = cameraService.state else {
            localState = .error(
                message: "Camera is not ready. On the iOS Simulator the camera does not produce real frames — run on hardware."
            )
            return
        }
        localState = .capturing
        do {
            let image = try await cameraService.captureOneFrame()
            let raw = try await classifierBridge.classify(image)
            let persisted = try predictionStore.record(raw: raw)
            lastPersisted = persisted
            localState = .showing(
                predictionId: persisted.id,
                label: persisted.label,
                parasitizedProb: persisted.parasitizedProb
            )
        } catch {
            localState = .error(message: error.localizedDescription)
        }
    }

    private func endSession() {
        cameraService.stop()
        localState = .idle
        lastPersisted = nil
    }
}
