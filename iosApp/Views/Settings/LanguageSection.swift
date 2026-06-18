import SwiftUI

/// Language picker section (spec §11). Editable, but the edit screen
/// triggers a fresh biometric prompt — spec §11 calls this out
/// explicitly to "prevent stranger-flips".
struct LanguageSection: View {

    @Environment(SettingsStore.self) private var settings

    var body: some View {
        Section {
            NavigationLink {
                EditLanguageView()
            } label: {
                HStack {
                    Text("Language")
                    Spacer()
                    Text(displayName(for: settings.language))
                        .foregroundStyle(.secondary)
                }
            }
        } header: {
            Text("Language")
        } footer: {
            Text("Changing language requires a fresh biometric prompt.")
        }
    }

    private func displayName(for raw: String) -> String {
        OnboardingLanguage(rawValue: raw)?.displayName ?? raw
    }
}
