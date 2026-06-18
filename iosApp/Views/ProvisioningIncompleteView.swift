import SwiftUI

/// Shown when the device has been provisioned by an admin but the
/// clinician profile has not yet been claimed. Phase 6 routes this
/// to the microscopist-onboarding flow. Per spec §10.
struct ProvisioningIncompleteView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Provisioning incomplete")
                .font(.title)
            Text("This device is awaiting clinician onboarding. Phase 6 wires the claim flow.")
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.leading, 24)
        .padding(.trailing, 24)
        .padding(.top, 24)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
