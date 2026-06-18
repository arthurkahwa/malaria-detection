import SwiftUI

/// The reset-device confirmation flow (spec §10 re-onboarding).
///
/// Order of operations:
///   1. Show the explanatory copy and a primary destructive button.
///   2. On tap, present a fresh biometric prompt via `AuthGate.unlock(...)`
///      (spec §9 lists Reset device as fresh-auth).
///   3. On success, surface a double-confirmation alert ("This will wipe…").
///   4. On final confirm, `ResetDeviceCoordinator.performReset()` wipes
///      the clinician row, writes `device_reprovisioned`, and flips
///      `OnboardingState.phase` back to `.adminProvisioning`. The
///      composition root auto-shows `OnboardingFlow` on next render.
struct ResetDeviceView: View {

    @Environment(AuthGate.self) private var authGate
    @Environment(ResetDeviceCoordinator.self) private var coordinator

    @State private var isAuthenticating: Bool = false
    @State private var showDoubleConfirm: Bool = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 12) {
                    Label("This action cannot be undone", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text("Resetting the device wipes the clinician profile and consent records on this device, then returns the app to admin provisioning. Predictions and audit history are preserved as chain-of-custody.")
                    Text("Phase 1 (admin provisioning) must be completed again on the next launch.")
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)
                .padding(.bottom, 4)
            }
            Section {
                Button(role: .destructive) {
                    Task { await startReset() }
                } label: {
                    HStack {
                        Spacer()
                        if isAuthenticating {
                            ProgressView()
                        } else {
                            Text("Reset device")
                                .fontWeight(.semibold)
                        }
                        Spacer()
                    }
                }
                .disabled(isAuthenticating)
            }
            if let errorText {
                Section { Text(errorText).foregroundStyle(.red) }
            }
        }
        .navigationTitle("Reset device")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Wipe clinician data?", isPresented: $showDoubleConfirm) {
            Button("Cancel", role: .cancel) { }
            Button("Wipe and re-provision", role: .destructive) {
                performWipe()
            }
        } message: {
            Text("This will wipe clinician data on this device. Predictions and audit history are preserved.")
        }
    }

    private func startReset() async {
        errorText = nil
        isAuthenticating = true
        defer { isAuthenticating = false }
        do {
            try await authGate.unlock(reason: "Confirm device reset")
            showDoubleConfirm = true
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func performWipe() {
        do {
            try coordinator.performReset()
        } catch {
            errorText = error.localizedDescription
        }
    }
}
