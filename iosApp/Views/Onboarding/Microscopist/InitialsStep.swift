import SwiftUI

/// Phase 2 step 2 (spec §10): optional 2-character initials or skip.
/// Initials appear next to the microscopist's identifier in audit
/// reports so reviewers can recognise who recorded an override.
struct InitialsStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var initials: String = ""
    @State private var errorMessage: String?

    private var trimmed: String {
        initials.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        WizardStepContainer(
            title: "Your initials",
            subtitle: "Up to two characters. These appear next to your overrides in the audit log. Optional — you can skip this.",
            stepIndicator: "Step 2 of 4",
            primary: .init(title: "Continue", isEnabled: trimmed.count <= 2) {
                save(initials: trimmed.isEmpty ? nil : trimmed)
            },
            secondary: .init(title: "Skip") {
                save(initials: nil)
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Initials")
                    .font(.headline)
                TextField("e.g. JM", text: $initials)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled(true)
                    .textInputAutocapitalization(.characters)
                    .submitLabel(.done)
                    .onChange(of: initials) { _, newValue in
                        // Clamp to 2 characters as the user types — the
                        // spec column is exactly 2 chars wide and longer
                        // strings would mis-render in the audit list.
                        if newValue.count > 2 {
                            initials = String(newValue.prefix(2))
                        }
                    }
            }
        }
        .alert(
            "Unable to save initials",
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

    private func save(initials: String?) {
        do {
            try onboarding.setMicroscopistInitials(initials)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
