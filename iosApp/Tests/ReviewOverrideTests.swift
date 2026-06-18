import SwiftData
import XCTest
@preconcurrency import Shared
@testable import MalariaDetector

/// Phase 9 review-override tests. Mirrors the Android-side
/// `ReviewOverrideStateTest` JVM cases so the contract is asserted
/// identically on both platforms (spec §12 review override).
///
/// The view itself is JVM/UI-agnostic for the state machine — the
/// `canSave` rule is exposed as a `static func` on `ReviewOverrideView`
/// so this test can exercise it without a SwiftUI runtime. The
/// audit-write roundtrip exercises `PredictionStore.override(...)` via
/// the in-memory fixture so the assertions cover the path the view
/// invokes on Save.
@MainActor
final class ReviewOverrideTests: XCTestCase {

    // MARK: - Save-enable state machine

    func testSaveEnable_requiresVerdictReasonAndCheckbox() {
        // No selection at all.
        XCTAssertFalse(ReviewOverrideView.canSave(
            verdict: nil, reason: nil, contextReviewed: false
        ))

        // Verdict only.
        XCTAssertFalse(ReviewOverrideView.canSave(
            verdict: .parasitized, reason: nil, contextReviewed: false
        ))

        // Verdict + reason but not reviewed.
        XCTAssertFalse(ReviewOverrideView.canSave(
            verdict: .uninfected,
            reason: Shared.OverrideReason.imageQuality,
            contextReviewed: false
        ))

        // Verdict + reviewed but no reason.
        XCTAssertFalse(ReviewOverrideView.canSave(
            verdict: .parasitized,
            reason: nil,
            contextReviewed: true
        ))

        // All three: enabled.
        XCTAssertTrue(ReviewOverrideView.canSave(
            verdict: .uninfected,
            reason: Shared.OverrideReason.modelFalsePositive,
            contextReviewed: true
        ))
    }

    // MARK: - Override roundtrip

    func testOverrideRoundtrip_setsFieldsAndWritesAuditEntry() throws {
        let fx = try TestSupport.makeFixture()
        let clinician = try fx.clinicianRepo.enroll(role: "microscopist", initials: "JM")
        try fx.clinicianRepo.markBiometricEnrolled(clinician)

        let store = PredictionStore(
            predictions: fx.predictionRepo,
            audit: fx.auditLog,
            clinician: fx.clinicianRepo
        )
        let prediction = TestSupport.samplePrediction(
            parasitizedProb: 0.53,
            label: "Parasitized",
            flaggedForReview: true
        )
        try fx.predictionRepo.insert(prediction)

        try store.override(
            prediction,
            verdict: "Uninfected",
            context: Shared.OverrideContext.review.canonical,
            reason: Shared.OverrideReason.modelFalsePositive.canonical,
            notes: "Edge artefact on the slide.",
            actorInitials: "AK",
            contextReviewed: true
        )

        // Prediction columns updated.
        let stored = try XCTUnwrap(fx.predictionRepo.byId(prediction.id))
        XCTAssertEqual(stored.clinicianOverride, "Uninfected")
        XCTAssertEqual(stored.overrideContext, "review")

        // Audit entry written with the full Phase 9 payload.
        let entries = try fx.auditRepo.entries(forAction: .overrideRecorded)
        XCTAssertEqual(entries.count, 1)
        let entry = try XCTUnwrap(entries.first)
        XCTAssertEqual(entry.resourceId, prediction.id)
        XCTAssertEqual(entry.overrideContext, "review")
        XCTAssertEqual(entry.overrideReason, "model_false_positive")
        XCTAssertEqual(entry.overrideNotes, "Edge artefact on the slide.")
        XCTAssertEqual(entry.overrideActorInitials, "AK")
        XCTAssertEqual(entry.contextReviewed, true)
        XCTAssertEqual(entry.actorId, clinician.actorId)
    }

    // MARK: - Idempotent retry

    /// Spec §12: "An override cannot be undone in v1." The PredictionDetail
    /// affordance hides itself once `clinicianOverride` is set, so a second
    /// override is unreachable from the UI — but the underlying write must
    /// remain safe if invoked (e.g. a stale screen in another nav stack).
    /// Each call appends a new audit entry; the displayed verdict reflects
    /// the most recent override only.
    func testOverride_secondCallAppendsAuditAndUpdatesVerdict() throws {
        let fx = try TestSupport.makeFixture()
        _ = try fx.clinicianRepo.enroll(role: "microscopist", initials: "JM")

        let store = PredictionStore(
            predictions: fx.predictionRepo,
            audit: fx.auditLog,
            clinician: fx.clinicianRepo
        )
        let prediction = TestSupport.samplePrediction(flaggedForReview: true)
        try fx.predictionRepo.insert(prediction)

        try store.override(
            prediction,
            verdict: "Uninfected",
            context: "review",
            reason: Shared.OverrideReason.imageQuality.canonical,
            notes: nil,
            actorInitials: "JM",
            contextReviewed: true
        )
        try store.override(
            prediction,
            verdict: "Parasitized",
            context: "review",
            reason: Shared.OverrideReason.other.canonical,
            notes: "Second reviewer disagrees.",
            actorInitials: "AK",
            contextReviewed: true
        )

        // Most-recent verdict on the row, two audit entries in the log.
        let stored = try XCTUnwrap(fx.predictionRepo.byId(prediction.id))
        XCTAssertEqual(stored.clinicianOverride, "Parasitized")
        XCTAssertEqual(stored.overrideContext, "review")
        let entries = try fx.auditRepo.entries(forAction: .overrideRecorded)
        XCTAssertEqual(entries.count, 2)
    }

    // MARK: - Display labels

    func testDisplayLabels_coverAllFiveCanonicalReasons() {
        // Spec §15 keeps these labels English even when the rest of the
        // app is localised; pinning the strings here means a translator
        // accidentally re-routing them through `.xcstrings` would trip a
        // test rather than silently shipping translated reasons.
        XCTAssertEqual(
            ReviewOverrideView.displayLabel(for: Shared.OverrideReason.imageQuality),
            "Image quality"
        )
        XCTAssertEqual(
            ReviewOverrideView.displayLabel(for: Shared.OverrideReason.atypicalMorphology),
            "Atypical morphology"
        )
        XCTAssertEqual(
            ReviewOverrideView.displayLabel(for: Shared.OverrideReason.modelFalsePositive),
            "Model false-positive"
        )
        XCTAssertEqual(
            ReviewOverrideView.displayLabel(for: Shared.OverrideReason.modelFalseNegative),
            "Model false-negative"
        )
        XCTAssertEqual(
            ReviewOverrideView.displayLabel(for: Shared.OverrideReason.other),
            "Other"
        )
    }
}
