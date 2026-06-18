import LocalAuthentication
import SwiftUI

/// Step 7 of admin provisioning (spec §10): biometric registration.
///
/// The "registration" happens implicitly: we call
/// `LAContext.evaluatePolicy(.deviceOwnerAuthentication, …)` and a
/// successful evaluation proves the operator has a usable biometric or
/// passcode on the device. Per spec §9 the same `LAContext` policy is
/// reused for routine unlock — no per-app biometric enrolment exists.
///
/// On success we call `OnboardingState.completeAdminProvisioning()` which
/// creates the admin `ClinicianProfile`, marks it biometricEnrolled, and
/// flips the phase to `.microscopistClaim`.
struct AdminBiometricStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var inFlight = false
    @State private var lastError: AuthError?

    var body: some View {
        WizardStepContainer(
            title: "Register administrator biometric",
            subtitle: "This authorizes admin-only actions like changing the threshold or wiping the device. It is not used for routine screening.",
            stepIndicator: "Step 7 of 8",
            primary: .init(title: inFlight ? "Verifying…" : "Register biometric", isEnabled: !inFlight) {
                runBiometric()
            }
        ) {
            VStack(alignment: .leading, spacing: 12) {
                Label("Tap the button below. Use Face ID, Touch ID, or your device passcode when prompted.", systemImage: "faceid")
                    .font(.body)
                Label("The biometric is registered with the operating system, not with this app. We never store the fingerprint or face data.", systemImage: "lock.shield")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
        }
        .alert(
            "Biometric registration failed",
            isPresented: Binding(
                get: { lastError != nil },
                set: { if !$0 { lastError = nil } }
            )
        ) {
            Button("OK", role: .cancel) { lastError = nil }
        } message: {
            Text(lastError?.errorDescription ?? "")
        }
    }

    private func runBiometric() {
        inFlight = true
        Task { @MainActor in
            defer { inFlight = false }
            let laContext = LAContext()
            var error: NSError?
            guard laContext.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error) else {
                lastError = .noBiometricsConfigured
                return
            }
            do {
                try await laContext.evaluatePolicy(
                    .deviceOwnerAuthentication,
                    localizedReason: "Register administrator biometric"
                )
            } catch let laError as LAError {
                switch laError.code {
                case .userCancel, .systemCancel, .appCancel:
                    lastError = .userCancelled
                default:
                    lastError = .authenticationFailed(reason: laError.localizedDescription)
                }
                return
            } catch {
                lastError = .authenticationFailed(reason: error.localizedDescription)
                return
            }

            do {
                _ = try onboarding.completeAdminProvisioning()
            } catch {
                lastError = .authenticationFailed(reason: error.localizedDescription)
            }
        }
    }
}
