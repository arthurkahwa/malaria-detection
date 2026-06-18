import SwiftData
import SwiftUI

/// Header stats + label editing + per-prediction list for one session.
/// Per spec §11 + §13. Predictions are queried via `@Query` filtered by
/// sessionId so the view stays reactive when new captures land.
struct SessionDetailView: View {

    let sessionId: String

    @Query private var predictions: [Prediction]

    @State private var showRelabel = false

    init(sessionId: String) {
        self.sessionId = sessionId
        _predictions = Query(
            filter: #Predicate<Prediction> { $0.sessionId == sessionId },
            sort: [SortDescriptor(\Prediction.timestamp, order: .forward)]
        )
    }

    private var stats: SessionStats? {
        SessionStats.from(predictions: predictions)
    }

    var body: some View {
        List {
            if let stats {
                Section("Overview") {
                    row(label: "Cells", value: "\(stats.count)")
                    row(label: "Parasitized", value: "\(stats.parasitizedCount)")
                    row(label: "Gray zone", value: "\(stats.grayZoneCount)")
                    row(label: "Mean parasitized prob.", value: stats.meanParasitizedFormatted)
                    row(label: "Date range", value: stats.dateRangeLabel)
                }

                Section("Label") {
                    HStack {
                        Text(stats.sessionLabel ?? "—")
                            .foregroundStyle(stats.sessionLabel == nil ? .secondary : .primary)
                        Spacer()
                        Button("Edit label") {
                            showRelabel = true
                        }
                    }
                }
            } else {
                Section {
                    Text("Session is empty")
                        .foregroundStyle(.secondary)
                }
            }

            Section("Predictions (\(predictions.count))") {
                ForEach(predictions) { p in
                    NavigationLink {
                        PredictionDetailView(prediction: p)
                    } label: {
                        PredictionRowView(prediction: p)
                    }
                }
            }
        }
        .navigationTitle("Session")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showRelabel) {
            NavigationStack {
                SessionRelabelView(sessionId: sessionId)
            }
        }
    }

    private func row(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
            Spacer()
            Text(value)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }
}
