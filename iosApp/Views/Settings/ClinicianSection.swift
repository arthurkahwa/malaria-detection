import SwiftData
import SwiftUI

/// Clinician profile section (spec §11).
///
/// - UUID: read-only, copyable. Long-press copies via the context menu.
/// - Role: read-only.
/// - Initials: editable — pushes [EditInitialsView] which triggers a
///   fresh biometric prompt before writing the change.
struct ClinicianSection: View {

    @Environment(\.modelContext) private var modelContext

    @State private var profile: ClinicianProfile?
    @State private var copyConfirmation: Bool = false

    var body: some View {
        Section {
            HStack {
                Text("UUID")
                Spacer()
                Text(profile?.actorId ?? "—")
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
            .contextMenu {
                Button {
                    if let id = profile?.actorId {
                        UIPasteboard.general.string = id
                        copyConfirmation = true
                    }
                } label: {
                    Label("Copy UUID", systemImage: "doc.on.doc")
                }
            }

            HStack {
                Text("Role")
                Spacer()
                Text(profile?.role.capitalized ?? "—")
                    .foregroundStyle(.secondary)
            }

            NavigationLink {
                EditInitialsView()
            } label: {
                HStack {
                    Text("Initials")
                    Spacer()
                    Text(profile?.initials ?? "—")
                        .foregroundStyle(.secondary)
                }
            }
        } header: {
            Text("Clinician profile")
        } footer: {
            Text(copyConfirmation ? "Copied!" : "UUID is the clinician's device-local identifier. It contains no personal data.")
        }
        .onAppear(perform: loadProfile)
    }

    private func loadProfile() {
        let repo = ClinicianRepository(context: modelContext)
        profile = try? repo.current()
    }
}
