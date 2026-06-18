import SwiftUI

/// Phase 2 step 1 (spec §10): welcome blurb that confirms the clinic
/// name this device was provisioned for. Tapping Continue calls
/// `startMicroscopistClaim()` which advances to the initials step.
struct MicroscopistWelcomeStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var errorMessage: String?

    private var clinicName: String {
        onboarding.pendingClinicName ?? "this clinic"
    }

    var body: some View {
        WizardStepContainer(
            title: "Welcome, microscopist",
            subtitle: nil,
            stepIndicator: "Step 1 of 4",
            primary: .init(title: "Continue") {
                do {
                    try onboarding.startMicroscopistClaim()
                } catch {
                    errorMessage = error.localizedDescription
                }
            }
        ) {
            Text("This device is provisioned for \(clinicName). The next few screens claim the device as yours.")
                .font(.body)
            Text("You'll register your own biometric so only you can unlock the app for routine screening.")
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .alert(
            "Unable to start microscopist claim",
            isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )
        ) {
            Button("OK", role: .cancel) { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
    }
}
