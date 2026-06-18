import SwiftData
import SwiftUI

/// Sessions list. Reactive query pulls every prediction; grouping happens
/// in Swift via `SessionStats.grouped`. v1.1 may move stats computation
/// into a shared Kotlin aggregator.
struct SessionsView: View {

    @Query(sort: \Prediction.timestamp, order: .reverse)
    private var allPredictions: [Prediction]

    private var sessions: [SessionStats] {
        SessionStats.grouped(allPredictions)
    }

    var body: some View {
        Group {
            if sessions.isEmpty {
                ContentUnavailableView(
                    "No sessions yet",
                    systemImage: "list.bullet.rectangle",
                    description: Text("Captured cells will be grouped into sessions here.")
                )
            } else {
                List {
                    ForEach(sessions) { session in
                        NavigationLink {
                            SessionDetailView(sessionId: session.sessionId)
                        } label: {
                            SessionRowView(stats: session)
                        }
                    }
                }
            }
        }
        .navigationTitle("Sessions")
    }
}
