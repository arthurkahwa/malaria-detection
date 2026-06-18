import SwiftUI

/// Clinic identity (read-only). Sourced from the `clinic_configured`
/// audit entry via `SettingsStore`. Read-only per spec §11: the clinic
/// config can only change via a full device reset (spec §10).
struct ClinicSection: View {

    @Environment(SettingsStore.self) private var settings

    var body: some View {
        Section {
            row("Name", value: settings.clinicName ?? "—")
            row("Jurisdiction", value: settings.jurisdiction ?? "—")
            row("Lawful basis", value: settings.lawfulBasis ?? "—")
        } header: {
            Text("Clinic")
        } footer: {
            Text("Clinic identity is set during admin provisioning and can only change via Reset device.")
        }
    }

    private func row(_ label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }
}
