import Foundation
import Observation
@preconcurrency import Shared

/// Write-path facade for `Prediction` rows. Views consume read state via
/// `@Query<Prediction>` directly (spec §6 "iOS: SwiftUI View with @Query");
/// this service exists for mutations and to bundle the matching audit
/// write next to every persistence change.
///
/// The active-screening flow calls `record(rawPrediction:threshold:)`
/// after the classifier returns. SessionGrouping is consulted to assign
/// the right sessionId.
@Observable
@MainActor
final class PredictionStore {
    private let predictions: PredictionRepository
    private let audit: AuditLog
    private let clinician: ClinicianRepository

    private(set) var lastError: Error?

    init(
        predictions: PredictionRepository,
        audit: AuditLog,
        clinician: ClinicianRepository
    ) {
        self.predictions = predictions
        self.audit = audit
        self.clinician = clinician
    }

    /// Persist a new prediction from the shared classifier's output.
    /// Computes label, flag, and session id via the shared helpers per
    /// spec §6 data flow.
    @discardableResult
    func record(
        raw: Shared.Prediction,
        threshold: Double = Double(Threshold.shared.DEFAULT)
    ) throws -> Prediction {
        let actor = try clinician.current()

        let parasitized = Double(raw.parasitizedProb)
        let label = Threshold.shared.label(
            parasitizedProb: raw.parasitizedProb,
            threshold: Float(threshold)
        )
        let flagged = Threshold.shared.shouldFlagForReview(parasitizedProb: raw.parasitizedProb)

        let priorTimestamp = (try? predictions.mostRecent())?.timestamp
        let priorSessionId = (try? predictions.mostRecent())?.sessionId
        let sessionId = SessionGrouping.shared.assignSessionId(
            previousPredictionTimestamp: priorTimestamp.map { kotlinInstant(from: $0) },
            previousSessionId: priorSessionId,
            now: kotlinInstant(from: Date())
        )

        let entity = Prediction(
            sessionId: sessionId,
            timestamp: Date(),
            modelId: raw.modelId,
            modelVersion: ModelHashCache.hash(for: raw.modelId),
            parasitizedProb: parasitized,
            uninfectedProb: Double(raw.uninfectedProb),
            label: label,
            threshold: threshold,
            flaggedForReview: flagged,
            inferenceMs: Int(raw.inferenceMs),
            imageHash: raw.imageHash,
            appVersion: BuildEnvironment.appVersion,
            osVersion: BuildEnvironment.osVersion
        )

        do {
            try predictions.insert(entity)
            audit.write(
                .predictionCreated,
                actorId: actor?.actorId ?? "unknown",
                actorRoleAtTime: actor?.role ?? "unknown",
                resourceType: "prediction",
                resourceId: entity.id,
                metadata: [
                    "model_id": entity.modelId,
                    "label": entity.label,
                    "flagged": String(entity.flaggedForReview),
                ]
            )
            lastError = nil
            return entity
        } catch {
            lastError = error
            throw error
        }
    }

    /// Record a clinician override. `context` is "live" or "review"
    /// (spec §12); the caller decides based on which UI surfaced it.
    func override(
        _ prediction: Prediction,
        verdict: String,
        context: String,
        reason: String,
        notes: String? = nil,
        actorInitials: String? = nil,
        contextReviewed: Bool? = nil
    ) throws {
        let actor = try clinician.current()
        try predictions.recordOverride(on: prediction, verdict: verdict, context: context)
        audit.write(
            .overrideRecorded,
            actorId: actor?.actorId ?? "unknown",
            actorRoleAtTime: actor?.role ?? "unknown",
            resourceType: "prediction",
            resourceId: prediction.id,
            metadata: ["verdict": verdict],
            overrideContext: context,
            overrideReason: reason,
            overrideNotes: notes,
            contextReviewed: contextReviewed,
            overrideActorInitials: actorInitials
        )
    }

    func markAsDuplicate(_ duplicate: Prediction, of originalId: String) throws {
        let actor = try clinician.current()
        try predictions.markAsDuplicate(duplicate, of: originalId)
        audit.write(
            .predictionLinkedAsDuplicate,
            actorId: actor?.actorId ?? "unknown",
            actorRoleAtTime: actor?.role ?? "unknown",
            resourceType: "prediction",
            resourceId: duplicate.id,
            metadata: ["original_id": originalId]
        )
    }

    func relabel(sessionId: String, label: String?) throws {
        let actor = try clinician.current()
        try predictions.relabel(sessionId: sessionId, label: label)
        audit.write(
            .sessionRelabeled,
            actorId: actor?.actorId ?? "unknown",
            actorRoleAtTime: actor?.role ?? "unknown",
            resourceType: "session",
            resourceId: sessionId,
            metadata: ["label": label ?? ""]
        )
    }
}

/// Bridge `Foundation.Date` ↔ `kotlinx.datetime.Instant`. The shared
/// `SessionGrouping` requires `Instant` arguments — we round-trip via
/// epoch milliseconds since both representations agree on UTC ms.
private func kotlinInstant(from date: Date) -> Kotlinx_datetimeInstant {
    Kotlinx_datetimeInstant.companion.fromEpochMilliseconds(
        epochMilliseconds: Int64(date.timeIntervalSince1970 * 1000)
    )
}
