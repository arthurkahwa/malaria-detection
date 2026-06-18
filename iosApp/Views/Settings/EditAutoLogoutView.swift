import SwiftUI

/// Edit the auto-logout timeout (admin only). Fresh biometric before
/// `auto_logout_changed` audit entry per spec §9 + §11.
struct EditAutoLogoutView: View {

    @Environment(\.dismiss) private var dismiss
    @Environment(AuthGate.self) private var authGate
    @Environment(SettingsStore.self) private var settings

    @State private var minutes: Int = 15
    @State private var isSaving: Bool = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section {
                Picker("Auto-logout", selection: $minutes) {
                    Text("5 minutes").tag(5)
                    Text("15 minutes").tag(15)
                    Text("30 minutes").tag(30)
                }
                .pickerStyle(.inline)
            } footer: {
                Text("The app locks itself after this much inactivity. Microscopists re-unlock with biometric.")
            }
            if let errorText {
                Section { Text(errorText).foregroundStyle(.red) }
            }
        }
        .navigationTitle("Auto-logout")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") {
                    Task { await save() }
                }
                .disabled(isSaving)
            }
        }
        .onAppear {
            minutes = settings.autoLogoutMinutes
        }
    }

    private func save() async {
        errorText = nil
        isSaving = true
        defer { isSaving = false }

        do {
            try await authGate.unlock(reason: "Confirm auto-logout change")
        } catch {
            errorText = error.localizedDescription
            return
        }

        settings.updateAutoLogoutMinutes(minutes)
        dismiss()
    }
}
