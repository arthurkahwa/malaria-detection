import SwiftUI

/// Crash logs section (spec §11 / §16). Phase 14 wires this to the
/// on-device `CrashLogStore`; the previous Phase 11 stub showed a
/// disabled placeholder.
struct CrashLogsSection: View {

    @Environment(CrashLogStore.self) private var store

    var body: some View {
        Section {
            HStack {
                Text("Crash log count")
                Spacer()
                Text("\(store.count())")
                    .foregroundStyle(.secondary)
            }
            NavigationLink {
                CrashLogsScreen()
            } label: {
                Label("Review and share", systemImage: "ladybug")
            }
        } header: {
            Text("Crash logs")
        } footer: {
            Text("If the app crashes, a diagnostic log is saved on this device only. Nothing is sent automatically. You can review and share individual logs above. Logs auto-expire after 30 days.")
        }
    }
}
