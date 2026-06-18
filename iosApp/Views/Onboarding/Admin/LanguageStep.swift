import SwiftUI

/// Step 1 of admin provisioning (spec §10): language picker.
///
/// Persists the chosen language to `UserDefaults` immediately on
/// selection — that way "Reset device" preserves it (spec §10
/// re-onboarding) without any extra plumbing.
struct LanguageStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var selectedLanguage: OnboardingLanguage = LanguagePreference.get()

    var body: some View {
        WizardStepContainer(
            title: "Choose language",
            subtitle: "Pick the language for the app. You can change this later from Settings.",
            stepIndicator: "Step 1 of 8",
            primary: .init(title: "Continue") {
                LanguagePreference.set(selectedLanguage)
                onboarding.advanceFromLanguage()
            }
        ) {
            Picker("Language", selection: $selectedLanguage) {
                ForEach(OnboardingLanguage.allCases) { language in
                    Text(language.displayName).tag(language)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
        }
    }
}
