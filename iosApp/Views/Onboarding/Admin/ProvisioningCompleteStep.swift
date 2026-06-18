import SwiftUI

/// Step 8 of admin provisioning (spec §10). The device is now in
/// `provisioned-unclaimed` state. The admin can either hand the device
/// to the microscopist (default path) or — for single-person deployments
/// — continue Phase 2 immediately.
struct ProvisioningCompleteStep: View {

    @Environment(OnboardingState.self) private var onboarding

    private var clinicName: String {
        onboarding.pendingClinicName ?? "your clinic"
    }

    var body: some View {
        WizardStepContainer(
            title: "Device provisioned",
            subtitle: "Phase 1 is complete.",
            stepIndicator: "Step 8 of 8",
            primary: .init(title: "Done — hand to microscopist") {
                onboarding.proceedToMicroscopistClaim()
            },
            secondary: .init(title: "You're the microscopist too? Continue Phase 2 now") {
                onboarding.proceedToMicroscopistClaim()
            }
        ) {
            VStack(alignment: .leading, spacing: 16) {
                Label("Device provisioned for \(clinicName).", systemImage: "checkmark.seal.fill")
                    .font(.title3)
                    .foregroundStyle(.green)

                Text("Hand this device to the microscopist who will use it. When they open the app they'll be walked through claiming the device and registering their own biometric.")
                    .font(.body)

                Text("Single-person deployment? Tap the alternate button below to continue Phase 2 right now.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
