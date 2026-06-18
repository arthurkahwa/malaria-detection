import Foundation
import Observation
@preconcurrency import Shared

/// Thin `@Observable` wrapper around the shared `Classifier`. Views
/// consume this directly so that future reactive state (in-flight
/// inference flag, last error, etc.) can be added without changing
/// the call site. Per spec §4 "iOS app — Example @Observable service
/// and view consumption."
@Observable
@MainActor
final class ClassifierBridge {

    private let classifier: Classifier

    init(classifier: Classifier) {
        self.classifier = classifier
    }

    /// Run a single image through the underlying Core ML classifier.
    /// Throws `InferenceError` (bridged from Kotlin) on failure.
    ///
    /// Returns `Shared.Prediction` (the raw DTO) — the SwiftData entity
    /// has the same short name so the return type is fully qualified.
    /// `PredictionStore.record(raw:)` maps the DTO to the @Model entity.
    func classify(_ image: ImageInput) async throws -> Shared.Prediction {
        try await classifier.classify(image: image)
    }
}
