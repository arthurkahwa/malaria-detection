import SwiftData
import SwiftUI

/// Newest-first list of stored predictions. Excludes duplicates by
/// default per spec §13; a toggle re-introduces them.
struct RecentPredictionsView: View {

    @Query(sort: \Prediction.timestamp, order: .reverse)
    private var all: [Prediction]

    @State private var showDuplicates: Bool = false

    private var visible: [Prediction] {
        showDuplicates ? all : all.filter { $0.duplicateOfId == nil }
    }

    var body: some View {
        List {
            Section {
                Toggle("Show duplicates", isOn: $showDuplicates)
            }

            if visible.isEmpty {
                Section {
                    Text("No predictions yet")
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 12)
                }
            } else {
                Section {
                    ForEach(visible) { prediction in
                        NavigationLink {
                            PredictionDetailView(prediction: prediction)
                        } label: {
                            PredictionRowView(prediction: prediction)
                        }
                    }
                }
            }
        }
        .navigationTitle("Recent predictions")
    }
}
