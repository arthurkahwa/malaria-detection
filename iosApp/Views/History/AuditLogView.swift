import SwiftData
import SwiftUI

/// Paginated audit log viewer with filters by action and date range.
/// Per spec §11: filter UI at the top, list paginated to most recent 200
/// for v1 (a "Load older" option lands in a later phase).
struct AuditLogView: View {

    @Query(sort: \AuditEntry.seq, order: .reverse)
    private var allEntries: [AuditEntry]

    @State private var selectedAction: AuditAction? = nil
    @State private var useDateFilter: Bool = false
    @State private var fromDate: Date = Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date()
    @State private var toDate: Date = Date()

    private var filtered: [AuditEntry] {
        var entries = allEntries
        if let action = selectedAction {
            let canonical = action.canonical
            entries = entries.filter { $0.action == canonical }
        }
        if useDateFilter {
            entries = entries.filter { $0.timestamp >= fromDate && $0.timestamp <= toDate }
        }
        return Array(entries.prefix(200))
    }

    var body: some View {
        List {
            Section("Filter") {
                Picker("Action", selection: $selectedAction) {
                    Text("All").tag(AuditAction?.none)
                    ForEach(AuditAction.allCases, id: \.self) { action in
                        Text(action.canonical).tag(AuditAction?.some(action))
                    }
                }
                Toggle("Date range", isOn: $useDateFilter)
                if useDateFilter {
                    DatePicker("From", selection: $fromDate, displayedComponents: [.date])
                    DatePicker("To", selection: $toDate, displayedComponents: [.date])
                }
            }

            if filtered.isEmpty {
                Section {
                    Text("No matching entries")
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 12)
                }
            } else {
                Section("Entries (\(filtered.count))") {
                    ForEach(filtered) { entry in
                        NavigationLink {
                            AuditEntryDetailView(entry: entry)
                        } label: {
                            AuditEntryRowView(entry: entry)
                        }
                    }
                }
            }
        }
        .navigationTitle("Audit log")
    }
}
