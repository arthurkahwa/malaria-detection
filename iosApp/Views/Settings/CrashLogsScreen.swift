import SwiftUI

/// Settings → Crash logs detail screen (spec §16).
///
/// Shows the on-device crash log list. Tap → action sheet → platform share
/// sheet (`UIActivityViewController`). Each share is audited as
/// `crash_log_shared` with the incident UUID.
struct CrashLogsScreen: View {

    @Environment(CrashLogStore.self) private var store

    @State private var pendingShare: CrashLogEntry?

    var body: some View {
        List {
            if store.entries.isEmpty {
                Section {
                    Text("No crash logs on this device.")
                        .foregroundStyle(.secondary)
                } footer: {
                    Text("If the app crashes, a diagnostic log is saved on this device only. Nothing is sent automatically.")
                }
            } else {
                Section {
                    ForEach(store.entries) { entry in
                        Button {
                            pendingShare = entry
                        } label: {
                            CrashLogRow(entry: entry)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.plain)
                    }
                } header: {
                    Text("\(store.entries.count) log\(store.entries.count == 1 ? "" : "s")")
                } footer: {
                    Text("Logs auto-expire after 30 days. Sharing is recorded in the audit log.")
                }
            }
        }
        .navigationTitle("Crash logs")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { store.refresh() }
        .sheet(item: $pendingShare) { entry in
            ShareSheet(activityItems: [store.shareableURL(entry)]) {
                store.didShare(entry)
                pendingShare = nil
            }
        }
    }
}

private struct CrashLogRow: View {
    let entry: CrashLogEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(formatAbsolute(entry.timestamp))
                .font(.body)
            HStack(spacing: 6) {
                Text(formatRelative(entry.timestamp))
                Text("·")
                Text(String(entry.incidentId.prefix(8)))
                    .font(.system(.caption, design: .monospaced))
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
    }

    private func formatAbsolute(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f.string(from: date)
    }

    private func formatRelative(_ date: Date) -> String {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .short
        return f.localizedString(for: date, relativeTo: Date())
    }
}

/// Wrapper around `UIActivityViewController` for SwiftUI. Spec §16: the
/// platform share sheet is the share mechanism on iOS.
private struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    let onCompletion: () -> Void

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: activityItems,
            applicationActivities: nil
        )
        controller.completionWithItemsHandler = { _, completed, _, _ in
            if completed { onCompletion() }
        }
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
