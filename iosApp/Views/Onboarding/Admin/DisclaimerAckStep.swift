import SwiftUI

/// Step 4 of admin provisioning (spec §10): medical-device disclaimer
/// acknowledgement (the repo `NOTICE` file).
struct DisclaimerAckStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var hasRead = false
    @State private var errorMessage: String?

    var body: some View {
        WizardStepContainer(
            title: "Medical-device disclaimer",
            subtitle: "Read the full disclaimer, then acknowledge to continue.",
            stepIndicator: "Step 4 of 8",
            primary: .init(title: "Continue", isEnabled: hasRead) {
                do {
                    try onboarding.acceptMedicalDisclaimer(version: LegalText.disclaimerVersion)
                } catch {
                    errorMessage = error.localizedDescription
                }
            }
        ) {
            Text(LegalText.disclaimerBody)
                .font(.callout)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Spec §16 onboarding disclosure: surfaces the on-device crash
            // log policy here so the user accepts it alongside the
            // medical-device disclaimer.
            VStack(alignment: .leading, spacing: 6) {
                Text("Diagnostic crash logs")
                    .font(.headline)
                Text("If the app crashes, a diagnostic log is saved on this device only. Nothing is sent automatically. You can review and share individual logs from Settings.")
                    .font(.callout)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            ConsentCheckbox(
                isChecked: $hasRead,
                label: "I understand this software is not a certified medical device and accept the disclaimer above."
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
