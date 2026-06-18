import SwiftData
import SwiftUI
@preconcurrency import Shared

/// Home tab: active screening. Per spec §11 + §19's "Example
/// @Observable service and view consumption." Views consume `@Query`
/// directly — no ViewModel intermediate (spec §6).
struct HomeTab: View {

    @Environment(AuthGate.self) private var authGate
    @Environment(PredictionStore.self) private var predictionStore

    // Newest-first reactive query of all stored predictions. SwiftData
    // updates the view whenever a new prediction is persisted.
    @Query(sort: \Prediction.timestamp, order: .reverse)
    private var recentPredictions: [Prediction]

    var body: some View {
        Group {
            switch authGate.state {
            case .locked:
                LockedPlaceholder()
            case .unlocked:
                ActiveScreeningView(recentPrediction: recentPredictions.first)
            case .provisionedUnclaimed:
                ProvisioningIncompleteView()
            }
        }
    }
}
