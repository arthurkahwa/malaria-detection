import SwiftUI

/// Step 2 of admin provisioning (spec §10): welcome blurb.
struct WelcomeStep: View {

    @Environment(OnboardingState.self) private var onboarding

    var body: some View {
        WizardStepContainer(
            title: "Welcome",
            subtitle: nil,
            stepIndicator: "Step 2 of 8",
            primary: .init(title: "Continue") {
                onboarding.advanceFromWelcome()
            }
        ) {
            Text("Malaria Detector is a decision-support tool for trained microscopists screening Giemsa-stained thin blood smears for Plasmodium parasites. It runs entirely on this device — no images leave the clinic.")
                .font(.body)
            Text("Setup happens in two phases. You (the clinic administrator) configure the device first, then hand it to a microscopist who claims it as their own.")
                .font(.body)
            Text("This app does not replace a trained microscopist. It produces a probability score; the microscopist decides.")
                .font(.body)
                .foregroundStyle(.secondary)
        }
    }
}
