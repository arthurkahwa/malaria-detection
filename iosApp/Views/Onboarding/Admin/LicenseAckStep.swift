import SwiftUI

/// Step 3 of admin provisioning (spec §10): Hippocratic License
/// acknowledgement.
struct LicenseAckStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var hasRead = false
    @State private var errorMessage: String?

    var body: some View {
        WizardStepContainer(
            title: "Hippocratic License",
            subtitle: "Read the full license, then acknowledge to continue.",
            stepIndicator: "Step 3 of 8",
            primary: .init(title: "Continue", isEnabled: hasRead) {
                do {
                    try onboarding.acceptHippocraticLicense(version: LegalText.licenseVersion)
                } catch {
                    errorMessage = error.localizedDescription
                }
            }
        ) {
            Text(LegalText.licenseBody)
                .font(.callout)
                .frame(maxWidth: .infinity, alignment: .leading)

            ConsentCheckbox(
                isChecked: $hasRead,
                label: "I have read and accept the Hippocratic License 3.0 (HL3-FULL)."
            )
        }
        .alert(
            "Unable to record acknowledgement",
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
