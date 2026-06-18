import SwiftData
import SwiftUI
@preconcurrency import Shared

/// AI Analysis detail view for a single prediction. Per spec §11 §12:
/// verdict + confidence, threshold at capture, model id, session link,
/// override status, and a duplicate-marking affordance. Writes an
/// audit `prediction_viewed` entry on appear.
struct PredictionDetailView: View {

    let prediction: Prediction

    @Environment(AuditLog.self) private var auditLog
    @Environment(\.modelContext) private var modelContext

    @State private var showMarkDuplicate = false
    @State private var didAudit = false
    @State private var copiedHash = false

    private var probabilityPercent: Int {
        Int((prediction.parasitizedProb * 100).rounded())
    }

    private var band: RiskBandIndicator.Band {
        RiskBandIndicator.band(for: prediction.parasitizedProb)
    }

    private var truncatedHash: String {
        String(prediction.imageHash.prefix(16)) + "…"
    }

    var body: some View {
        List {
            verdictSection
            contextSection
            sessionSection
            actionsSection
        }
        .navigationTitle("AI Analysis")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showMarkDuplicate) {
            NavigationStack {
                MarkAsDuplicateView(duplicate: prediction)
            }
        }
        .onAppear {
            recordViewedOnce()
        }
    }

    // MARK: - Sections

    private var verdictSection: some View {
        Section("Verdict") {
            HStack(alignment: .firstTextBaseline) {
                Text(prediction.label)
                    .font(.title2.weight(.semibold))
                Spacer()
                Text("\(probabilityPercent)%")
                    .font(.title3.monospacedDigit())
            }
            RiskBandIndicator(parasitizedProb: prediction.parasitizedProb)
            if let override = prediction.clinicianOverride {
                let context = prediction.overrideContext ?? "review"
                Text("Originally \(prediction.label) \(probabilityPercent)%, overridden to \(override) (in \(context))")
                    .font(.callout)
                    .foregroundStyle(.purple)
            }
        }
    }

    private var contextSection: some View {
        Section("Context") {
            row(label: "Captured (UTC)", value: prediction.timestamp.iso8601Utc)
            row(label: "Captured (local)", value: prediction.timestamp.formatted(date: .abbreviated, time: .standard))
            row(label: "Model", value: prediction.modelId)
            row(label: "Model version", value: prediction.modelVersion)
            row(label: "Threshold at capture", value: String(format: "%.2f", prediction.threshold))
            row(label: "Inference time", value: "\(prediction.inferenceMs) ms")
            Button {
                #if canImport(UIKit)
                UIPasteboard.general.string = prediction.imageHash
                #endif
                copiedHash = true
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 1_500_000_000)
                    copiedHash = false
                }
            } label: {
                HStack {
                    Text("Image hash")
                        .foregroundStyle(.primary)
                    Spacer()
                    Text(copiedHash ? "Copied!" : truncatedHash)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            .accessibilityLabel("Image hash. Tap to copy the full SHA-256.")
        }
    }

    private var sessionSection: some View {
        Section("Session") {
            if let label = prediction.sessionLabel, !label.isEmpty {
                row(label: "Label", value: label)
            }
            NavigationLink {
                SessionDetailView(sessionId: prediction.sessionId)
            } label: {
                HStack {
                    Text("Open session")
                    Spacer()
                    Text(String(prediction.sessionId.prefix(8)))
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var actionsSection: some View {
        Section("Actions") {
            Button {
                showMarkDuplicate = true
            } label: {
                Label("Mark as duplicate of…", systemImage: "doc.on.doc")
            }
            .disabled(prediction.duplicateOfId != nil)

            // Spec §12: "An override cannot be undone in v1". Hide the
            // affordance once an override has been recorded so the UI
            // doesn't suggest a second pass is meaningful — re-overrides
            // would write a new audit entry but the displayed verdict
            // only ever reflects the most recent override.
            if prediction.clinicianOverride == nil {
                NavigationLink {
                    ReviewOverrideView(prediction: prediction)
                } label: {
                    Label("Review and override", systemImage: "pencil.and.outline")
                }
            }
        }
    }

    private func row(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(.primary)
            Spacer()
            Text(value)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }

    // MARK: - Audit

    /// Writes `prediction_viewed` exactly once per view appearance. We
    /// guard with a flag so navigating back-and-forth in the same
    /// `NavigationStack` instance doesn't multiply entries — `onAppear`
    /// fires every time the view returns to screen. This matches spec
    /// §8's audit-action vocabulary expectation: one user-initiated
    /// "open detail" event per row tap.
    private func recordViewedOnce() {
        guard !didAudit else { return }
        didAudit = true
        let actor = try? ClinicianRepository(context: modelContext).current()
        auditLog.write(
            .predictionViewed,
            actorId: actor?.actorId ?? "unknown",
            actorRoleAtTime: actor?.role ?? "unknown",
            resourceType: "prediction",
            resourceId: prediction.id
        )
    }
}

private extension Date {
    var iso8601Utc: String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.string(from: self)
    }
}
