import SwiftUI
@preconcurrency import Shared

/// Step 6 of admin provisioning (spec §10): inference policy — threshold,
/// default model, auto-logout timeout.
struct InferencePolicyStep: View {

    @Environment(OnboardingState.self) private var onboarding
    @Environment(ModelRegistryService.self) private var modelRegistry

    @State private var threshold: Double = 0.3
    @State private var selectedModelId: String = "BNLeaky_Keras"
    @State private var autoLogoutMinutes: Int = 15

    private var entries: [ModelRegistryEntry] {
        modelRegistry.registry.all()
    }

    var body: some View {
        WizardStepContainer(
            title: "Inference policy",
            subtitle: "These choices apply to every prediction this device makes. Admins can adjust them later from Settings.",
            stepIndicator: "Step 6 of 8",
            primary: .init(title: "Continue") {
                onboarding.setInferencePolicy(
                    threshold: threshold,
                    defaultModelId: selectedModelId,
                    autoLogoutMinutes: autoLogoutMinutes
                )
            }
        ) {
            ThresholdSlider(threshold: $threshold)

            VStack(alignment: .leading, spacing: 8) {
                Text("Default model")
                    .font(.headline)
                ModelPicker(selectedId: $selectedModelId, entries: entries)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Auto-logout")
                    .font(.headline)
                Picker("Auto-logout", selection: $autoLogoutMinutes) {
                    Text("5 minutes").tag(5)
                    Text("15 minutes").tag(15)
                    Text("30 minutes").tag(30)
                }
                .pickerStyle(.segmented)
                Text("The app locks itself after this much inactivity. Microscopists re-unlock with biometric.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear {
            // Defensive default: the registry might omit the bundled
            // model id if `model_registry.json` is mid-migration.
            if entries.contains(where: { $0.id == "BNLeaky_Keras" }) {
                selectedModelId = "BNLeaky_Keras"
            } else if let first = entries.first {
                selectedModelId = first.id
            }
        }
    }
}
