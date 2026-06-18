import SwiftUI
@preconcurrency import Shared

/// Step 5 of admin provisioning (spec §10): clinic name + jurisdiction +
/// lawful basis.
struct ClinicDetailsStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var clinicName: String = ""
    @State private var jurisdiction: Jurisdiction = .other
    @State private var lawfulBasis: LawfulBasis = .healthProvision
    @State private var errorMessage: String?

    private var trimmedName: String {
        clinicName.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        WizardStepContainer(
            title: "Clinic details",
            subtitle: "These appear on exports and in the audit log. You can change them later in Settings (admin biometric required).",
            stepIndicator: "Step 5 of 8",
            primary: .init(title: "Continue", isEnabled: !trimmedName.isEmpty) {
                do {
                    try onboarding.configureClinic(
                        name: trimmedName,
                        jurisdiction: jurisdiction.canonical,
                        lawfulBasis: lawfulBasis.canonical,
                        version: "v1.0"
                    )
                } catch {
                    errorMessage = error.localizedDescription
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Clinic name")
                    .font(.headline)
                TextField("e.g. Kisumu District Health Centre", text: $clinicName)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled(false)
                    .submitLabel(.done)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Jurisdiction")
                    .font(.headline)
                JurisdictionPicker(selection: $jurisdiction)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Lawful basis")
                    .font(.headline)
                LawfulBasisPicker(selection: $lawfulBasis)
            }
        }
        .alert(
            "Unable to save clinic details",
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
