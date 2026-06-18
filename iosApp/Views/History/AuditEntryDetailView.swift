import SwiftUI

/// Full breakdown of one AuditEntry. Per spec §11 the metadataJson is
/// pretty-printed as key-value rows; for OVERRIDE_RECORDED entries the
/// override-specific fields surface in a dedicated section.
struct AuditEntryDetailView: View {

    let entry: AuditEntry

    private var metadataRows: [(String, String)] {
        guard let data = entry.metadataJson.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data, options: []),
              let dict = object as? [String: Any] else {
            return []
        }
        return dict.keys.sorted().map { key in
            (key, String(describing: dict[key] ?? ""))
        }
    }

    private var isOverrideEntry: Bool {
        entry.action == AuditAction.overrideRecorded.canonical
    }

    var body: some View {
        List {
            Section("Action") {
                row(label: "Action", value: entry.action)
                row(label: "Sequence", value: "#\(entry.seq)")
                row(label: "Timestamp (UTC)", value: entry.timestamp.iso8601Utc)
                row(label: "Timestamp (local)", value: entry.timestamp.formatted(date: .abbreviated, time: .standard))
            }

            Section("Actor") {
                row(label: "Actor ID", value: entry.actorId, mono: true)
                row(label: "Role at time", value: entry.actorRoleAtTime)
            }

            Section("Resource") {
                row(label: "Type", value: entry.resourceType ?? "—")
                row(label: "ID", value: entry.resourceId ?? "—", mono: entry.resourceId != nil)
            }

            Section("Environment") {
                row(label: "App version", value: entry.appVersion)
                row(label: "OS version", value: entry.osVersion)
            }

            if isOverrideEntry {
                Section("Override") {
                    row(label: "Context", value: entry.overrideContext ?? "—")
                    row(label: "Reason", value: entry.overrideReason ?? "—")
                    row(label: "Notes", value: entry.overrideNotes ?? "—")
                    row(label: "Context reviewed", value: entry.contextReviewed.map { $0 ? "Yes" : "No" } ?? "—")
                    row(label: "Override initials", value: entry.overrideActorInitials ?? "—")
                }
            }

            Section("Metadata") {
                if metadataRows.isEmpty {
                    Text("(empty)")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(metadataRows, id: \.0) { key, value in
                        HStack(alignment: .firstTextBaseline) {
                            Text(key)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(value)
                                .font(.callout)
                                .multilineTextAlignment(.trailing)
                        }
                    }
                }
            }
        }
        .navigationTitle("Audit entry")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func row(label: String, value: String, mono: Bool = false) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
            Spacer()
            Text(value)
                .font(mono ? .caption.monospaced() : .callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }
}

private extension Date {
    var iso8601Utc: String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.string(from: self)
    }
}
