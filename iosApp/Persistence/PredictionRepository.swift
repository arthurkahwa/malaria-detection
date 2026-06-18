import Foundation
import SwiftData

/// Persists and queries `Prediction` rows. All operations run on the main
/// actor because SwiftData's `ModelContext` is main-actor-isolated in the
/// production composition root.
@MainActor
struct PredictionRepository {
    let context: ModelContext

    /// Insert a new prediction. Caller is responsible for computing
    /// `label`, `flaggedForReview`, and `sessionId` via the shared
    /// `Threshold` / `SessionGrouping` helpers before constructing the
    /// entity. Returns the persisted entity.
    @discardableResult
    func insert(_ prediction: Prediction) throws -> Prediction {
        context.insert(prediction)
        try context.save()
        return prediction
    }

    /// Newest-first fetch, optionally limited.
    func recent(limit: Int = 50) throws -> [Prediction] {
        var descriptor = FetchDescriptor<Prediction>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        descriptor.fetchLimit = limit
        return try context.fetch(descriptor)
    }

    /// All predictions in a given session, oldest-first (the order a
    /// clinician would have captured them).
    func inSession(_ sessionId: String) throws -> [Prediction] {
        let descriptor = FetchDescriptor<Prediction>(
            predicate: #Predicate { $0.sessionId == sessionId },
            sortBy: [SortDescriptor(\.timestamp, order: .forward)]
        )
        return try context.fetch(descriptor)
    }

    /// Predictions in the gray zone awaiting clinician review.
    /// Excludes those already overridden.
    func flaggedForReview() throws -> [Prediction] {
        let descriptor = FetchDescriptor<Prediction>(
            predicate: #Predicate {
                $0.flaggedForReview == true && $0.clinicianOverride == nil
            },
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    func byId(_ id: String) throws -> Prediction? {
        let descriptor = FetchDescriptor<Prediction>(
            predicate: #Predicate { $0.id == id }
        )
        return try context.fetch(descriptor).first
    }

    /// Most recent prediction (used by SessionGrouping to decide whether
    /// the next capture starts a new session).
    func mostRecent() throws -> Prediction? {
        var descriptor = FetchDescriptor<Prediction>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        descriptor.fetchLimit = 1
        return try context.fetch(descriptor).first
    }

    /// Records a clinician override on an existing prediction. Both `live`
    /// and `review` contexts go through here; the caller picks the context
    /// value per spec §12.
    func recordOverride(
        on prediction: Prediction,
        verdict: String,
        context overrideContext: String
    ) throws {
        prediction.clinicianOverride = verdict
        prediction.overrideContext = overrideContext
        try self.context.save()
    }

    /// Marks a prediction as a duplicate of an earlier one. The duplicate
    /// stays in storage (no hard delete in v1 per spec §13).
    func markAsDuplicate(_ duplicate: Prediction, of originalId: String) throws {
        duplicate.duplicateOfId = originalId
        try context.save()
    }

    /// Sets the free-text session label on every prediction in the
    /// session (denormalised for query simplicity per spec §13).
    func relabel(sessionId: String, label: String?) throws {
        let predictions = try inSession(sessionId)
        for p in predictions {
            p.sessionLabel = label
        }
        try context.save()
    }
}
