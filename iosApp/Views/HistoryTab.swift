import SwiftUI

/// History tab: predictions, sessions, audit log, data management.
/// Per spec §11. Gated by `AuthGate` (same lock model as `HomeTab`).
struct HistoryTab: View {

    @Environment(AuthGate.self) private var authGate

    var body: some View {
        Group {
            switch authGate.state {
            case .locked:
                LockedPlaceholder()
            case .unlocked:
                HistoryRoot()
            case .provisionedUnclaimed:
                ProvisioningIncompleteView()
            }
        }
    }
}

/// The unlocked content: navigation root listing the five subsections in
/// the order spec §11 prescribes.
private struct HistoryRoot: View {
    var body: some View {
        NavigationStack {
            List {
                NavigationLink {
                    RecentPredictionsView()
                } label: {
                    Label("Recent predictions", systemImage: "clock.arrow.circlepath")
                }
                NavigationLink {
                    FlaggedForReviewView()
                } label: {
                    Label("Flagged for review", systemImage: "flag.fill")
                }
                NavigationLink {
                    SessionsView()
                } label: {
                    Label("Sessions", systemImage: "list.bullet.rectangle")
                }
                NavigationLink {
                    AuditLogView()
                } label: {
                    Label("Audit log", systemImage: "doc.text.magnifyingglass")
                }
                NavigationLink {
                    DataManagementView()
                } label: {
                    Label("Data management", systemImage: "externaldrive")
                }
            }
            .navigationTitle("History")
        }
    }
}
