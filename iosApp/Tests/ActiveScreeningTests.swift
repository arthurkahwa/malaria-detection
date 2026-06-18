import SwiftData
import XCTest
@preconcurrency import Shared
@testable import MalariaDetector

/// Phase 8 + Phase 9 live-override tests. These exercise the pure
/// state machine and the `PredictionStore.override(... context: "live")`
/// write path. The camera path itself is **not** exercised — the iOS
/// Simulator's `AVCaptureSession` does not produce real frames, so any
/// hardware-dependent assertion would only flake. The Capture flow is
/// validated end-to-end on real iPhone hardware, documented in
/// `ActiveScreeningView`'s header comment.
@MainActor
final class ActiveScreeningTests: XCTestCase {

    // MARK: - Live override roundtrip (Phase 9 live half)

    func testLiveOverride_setsRowAndWritesAuditEntry_withNullContextReviewed() throws {
        let fx = try TestSupport.makeFixture()
        let clinician = try fx.clinicianRepo.enroll(role: "microscopist", initials: "JM")
        try fx.clinicianRepo.markBiometricEnrolled(clinician)

        let store = PredictionStore(
            predictions: fx.predictionRepo,
            audit: fx.auditLog,
            clinician: fx.clinicianRepo
        )
        let prediction = TestSupport.samplePrediction(
            parasitizedProb: 0.87,
            label: "Parasitized",
            flaggedForReview: false
        )
        try fx.predictionRepo.insert(prediction)

        // Mirror what `LiveOverrideSheet.commit(...)` calls into.
        try store.override(
            prediction,
            verdict: "Uninfected",
            context: Shared.OverrideContext.live.canonical,
            reason: Shared.OverrideReason.modelFalsePositive.canonical,
            notes: nil,
            actorInitials: nil,
            contextReviewed: nil
        )

        // Prediction columns updated to the live override.
        let stored = try XCTUnwrap(fx.predictionRepo.byId(prediction.id))
        XCTAssertEqual(stored.clinicianOverride, "Uninfected")
        XCTAssertEqual(stored.overrideContext, "live")

        // Audit entry written per spec §12 live override semantics:
        // overrideContext = "live", overrideReason populated,
        // overrideActorInitials = nil, overrideNotes = nil,
        // contextReviewed = nil.
        let entries = try fx.auditRepo.entries(forAction: .overrideRecorded)
        XCTAssertEqual(entries.count, 1)
        let entry = try XCTUnwrap(entries.first)
        XCTAssertEqual(entry.resourceId, prediction.id)
        XCTAssertEqual(entry.overrideContext, "live")
        XCTAssertEqual(entry.overrideReason, "model_false_positive")
        XCTAssertNil(entry.overrideNotes)
        XCTAssertNil(entry.overrideActorInitials)
        XCTAssertNil(entry.contextReviewed)
        XCTAssertEqual(entry.actorId, clinician.actorId)
    }

    // MARK: - Canonical reason mapping

    func testOverrideReason_canonicalStrings_areLowercaseSnake() {
        // Spec §5 mandates the canonical wire form for every override
        // reason — lowercase snake_case, stable across the audit log,
        // exports, and v2 readers. Pin them here so a future Kotlin
        // refactor that accidentally renames them trips the build.
        XCTAssertEqual(Shared.OverrideReason.imageQuality.canonical, "image_quality")
        XCTAssertEqual(Shared.OverrideReason.atypicalMorphology.canonical, "atypical_morphology")
        XCTAssertEqual(Shared.OverrideReason.modelFalsePositive.canonical, "model_false_positive")
        XCTAssertEqual(Shared.OverrideReason.modelFalseNegative.canonical, "model_false_negative")
        XCTAssertEqual(Shared.OverrideReason.other.canonical, "other")
        XCTAssertEqual(Shared.OverrideContext.live.canonical, "live")
        XCTAssertEqual(Shared.OverrideContext.review.canonical, "review")
    }

    // MARK: - Permission-denied user-facing message

    func testPermissionDenied_errorDescription_isNonEmptyUserFacing() throws {
        // The string is what the permission-denied fallback view shows
        // when the system refuses access; the message must mention how
        // to grant permission so a user without read-the-source context
        // can recover. Pin the recovery hint here.
        let message = try XCTUnwrap(CameraService.CameraError.permissionDenied.errorDescription)
        XCTAssertFalse(message.isEmpty)
        XCTAssertTrue(message.localizedCaseInsensitiveContains("settings"))
    }
}
