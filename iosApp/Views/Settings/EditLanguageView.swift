import SwiftUI

/// Edit the UI language. Fresh biometric required to prevent
/// stranger-flips per spec §11.
struct EditLanguageView: View {

    @Environment(\.dismiss) private var dismiss
    @Environment(AuthGate.self) private var authGate
    @Environment(SettingsStore.self) private var settings

    @State private var selection: OnboardingLanguage = .english
    @State private var isSaving: Bool = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section {
                ForEach(OnboardingLanguage.allCases) { lang in
                    Button {
                        selection = lang
                    } label: {
                        HStack {
                            Text(lang.displayName)
                            Spacer()
                            if selection == lang {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                    .foregroundStyle(.primary)
                }
            } header: {
                Text("Language")
            } footer: {
                Text("Only English is fully translated in v1. Other locales fall back to English strings.")
            }
            if let errorText {
                Section { Text(errorText).foregroundStyle(.red) }
            }
        }
        .navigationTitle("Language")
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
            selection = OnboardingLanguage(rawValue: settings.language) ?? .english
        }
    }

    private func save() async {
        errorText = nil
        isSaving = true
        defer { isSaving = false }

        do {
            try await authGate.unlock(reason: "Confirm language change")
        } catch {
            errorText = error.localizedDescription
            return
        }

        settings.updateLanguage(selection)
        dismiss()
    }
}
