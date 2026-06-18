import LocalAuthentication
import SwiftUI

/// Phase 2 step 3 (spec §10): microscopist biometric registration.
/// Mirrors `AdminBiometricStep` but for the routine-unlock biometric
/// rather than the admin authorisation biometric. The OS evaluates the
/// same `deviceOwnerAuthentication` policy; the role check is what
/// differentiates the two contexts going forward (spec §10 single-person
/// deployment).
struct MicroscopistBiometricStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var inFlight = false
    @State private var lastError: AuthError?

    var body: some View {
        WizardStepContainer(
            title: "Register your biometric",
            subtitle: "Use Face ID, Touch ID, or your device passcode. This is how you'll unlock the app each session.",
            stepIndicator: "Step 3 of 4",
            primary: .init(title: inFlight ? "Verifying…" : "Register biometric", isEnabled: !inFlight) {
                runBiometric()
            }
        ) {
            VStack(alignment: .leading, spacing: 12) {
                Label("The biometric is registered with the operating system. The app never sees your face or fingerprint data.", systemImage: "lock.shield")
                    .font(.body)
                    .foregroundStyle(.secondary)
                Label("After this step you'll see a short orientation, then the app is ready for screening.", systemImage: "checkmark.circle")
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
                    localizedReason: "Register microscopist biometric"
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
                try onboarding.completeMicroscopistClaim()
            } catch {
                lastError = .authenticationFailed(reason: error.localizedDescription)
            }
        }
    }
}
