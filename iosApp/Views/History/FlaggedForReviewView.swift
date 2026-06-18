import SwiftData
import SwiftUI

/// Predictions flagged for clinician review and not yet overridden.
/// Predicate mirrors `PredictionRepository.flaggedForReview()` exactly
/// (spec §11).
struct FlaggedForReviewView: View {

    @Query(
        filter: #Predicate<Prediction> {
            $0.flaggedForReview == true && $0.clinicianOverride == nil
        },
        sort: \Prediction.timestamp,
        order: .reverse
    )
    private var flagged: [Prediction]

    var body: some View {
        Group {
            if flagged.isEmpty {
                ContentUnavailableView(
                    "Nothing flagged",
                    systemImage: "checkmark.seal",
                    description: Text("Gray-zone predictions awaiting review will appear here.")
                )
            } else {
                List {
                    ForEach(flagged) { prediction in
                        NavigationLink {
                            PredictionDetailView(prediction: prediction)
                        } label: {
                            PredictionRowView(prediction: prediction)
                        }
                    }
                }
            }
        }
        .navigationTitle("Flagged for review")
    }
}
