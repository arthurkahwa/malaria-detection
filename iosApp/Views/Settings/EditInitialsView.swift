import SwiftData
import SwiftUI

/// Edit microscopist initials. Triggers a fresh biometric prompt before
/// writing per spec §9 (every Settings edit is fresh-auth — this isn't
/// strictly an admin-only action but the spec partials list initials as
/// editable→biometric to keep the partial-lock model simple).
struct EditInitialsView: View {

    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext
    @Environment(AuthGate.self) private var authGate
    @Environment(SettingsStore.self) private var settings

    @State private var initials: String = ""
    @State private var isSaving: Bool = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section {
                TextField("Initials", text: $initials)
                    .textInputAutocapitalization(.characters)
                    .autocorrectionDisabled(true)
                    .onChange(of: initials) { _, newValue in
                        if newValue.count > 2 {
                            initials = String(newValue.prefix(2))
                        }
                    }
            } footer: {
                Text("Up to two characters. Appears next to overrides in the audit log.")
            }
            if let errorText {
                Section { Text(errorText).foregroundStyle(.red) }
            }
        }
        .navigationTitle("Edit initials")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") {
                    Task { await save() }
                }
                .disabled(isSaving)
            }
        }
        .onAppear(perform: loadInitials)
    }

    private func loadInitials() {
        let repo = ClinicianRepository(context: modelContext)
        initials = (try? repo.current())?.initials ?? ""
    }

    private func save() async {
        errorText = nil
        isSaving = true
        defer { isSaving = false }

        do {
            try await authGate.unlock(reason: "Confirm initials change")
        } catch {
            errorText = error.localizedDescription
            return
        }

        do {
            let trimmed = initials.trimmingCharacters(in: .whitespaces)
            try settings.updateInitials(trimmed.isEmpty ? nil : trimmed)
            dismiss()
        } catch {
            errorText = error.localizedDescription
        }
    }
}
