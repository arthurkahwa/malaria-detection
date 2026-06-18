import SwiftUI

/// Inference policy section (spec §11). Threshold / default model / auto-
/// logout are editable for admins (fresh biometric per spec §9) and
/// read-only for microscopists.
struct InferenceSection: View {

    @Environment(SettingsStore.self) private var settings

    @State private var role: String = ""

    private var isAdmin: Bool { role == "admin" }

    var body: some View {
        Section {
            if isAdmin {
                NavigationLink {
                    EditThresholdView()
                } label: {
                    valueRow("Decision threshold", value: String(format: "%.2f", settings.threshold))
                }
                NavigationLink {
                    EditDefaultModelView()
                } label: {
                    valueRow("Default model", value: settings.defaultModelId)
                }
                NavigationLink {
                    EditAutoLogoutView()
                } label: {
                    valueRow("Auto-logout", value: "\(settings.autoLogoutMinutes) min")
                }
            } else {
                valueRow("Decision threshold", value: String(format: "%.2f", settings.threshold))
                valueRow("Default model", value: settings.defaultModelId)
                valueRow("Auto-logout", value: "\(settings.autoLogoutMinutes) min")
            }
        } header: {
            Text("Inference")
        } footer: {
            Text(isAdmin
                 ? "Editing any value requires a fresh biometric prompt."
                 : "Only the device administrator can change inference policy.")
        }
        .onAppear(perform: loadRole)
    }

    private func valueRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
        }
    }

    @Environment(\.modelContext) private var modelContext

    private func loadRole() {
        let repo = ClinicianRepository(context: modelContext)
        role = (try? repo.current())?.role ?? ""
    }
}
