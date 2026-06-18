import SwiftData
import XCTest
@testable import MalariaDetector
@preconcurrency import Shared

/// Phase 10 history-tab tests. Covers session grouping, flagged-for-review
/// filtering, risk-band classification, and the ASCII-only relabel
/// validator (spec §11 + §13).
@MainActor
final class HistoryTabTests: XCTestCase {

    // MARK: - Sessions grouping

    func testSessionsGrouping_shapeAndCounts() throws {
        let fx = try TestSupport.makeFixture()
        let sessionA = UUID().uuidString
        let sessionB = UUID().uuidString

        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionA, timestamp: Date(timeIntervalSince1970: 1000)))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionA, timestamp: Date(timeIntervalSince1970: 2000)))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionA, timestamp: Date(timeIntervalSince1970: 3000)))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionB, timestamp: Date(timeIntervalSince1970: 5000)))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionB, timestamp: Date(timeIntervalSince1970: 6000)))

        let all = try fx.predictionRepo.recent(limit: 100)
        let groups = SessionStats.grouped(all)

        XCTAssertEqual(groups.count, 2)
        let byId = Dictionary(uniqueKeysWithValues: groups.map { ($0.sessionId, $0) })
        XCTAssertEqual(byId[sessionA]?.count, 3)
        XCTAssertEqual(byId[sessionB]?.count, 2)
        XCTAssertEqual(Set(byId.keys), Set([sessionA, sessionB]))

        // Session B has the latest timestamp → sorted first.
        XCTAssertEqual(groups.first?.sessionId, sessionB)
    }

    func testSessionsGrouping_grayZoneAndParasitizedCounts() throws {
        let fx = try TestSupport.makeFixture()
        let session = UUID().uuidString

        // Probabilities chosen to land in each band:
        //   0.20 → low (Uninfected), 0.50 → gray zone (Parasitized at default 0.3 threshold),
        //   0.90 → high (Parasitized), 0.10 → low (Uninfected).
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: session, parasitizedProb: 0.20, label: "Uninfected", threshold: 0.3))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: session, parasitizedProb: 0.50, label: "Parasitized", threshold: 0.3))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: session, parasitizedProb: 0.90, label: "Parasitized", threshold: 0.3))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: session, parasitizedProb: 0.10, label: "Uninfected", threshold: 0.3))

        let inSession = try fx.predictionRepo.inSession(session)
        guard let stats = SessionStats.from(predictions: inSession) else {
            XCTFail("expected stats")
            return
        }

        XCTAssertEqual(stats.count, 4)
        XCTAssertEqual(stats.parasitizedCount, 2)
        XCTAssertEqual(stats.grayZoneCount, 1)
        XCTAssertEqual(stats.meanParasitizedProb, (0.20 + 0.50 + 0.90 + 0.10) / 4.0, accuracy: 0.0001)
    }

    // MARK: - Flagged-for-review filter

    func testFlaggedForReview_excludesOverriddenAndUnflagged() throws {
        let fx = try TestSupport.makeFixture()

        // Five predictions: only two should surface — flagged AND not overridden.
        try fx.predictionRepo.insert(TestSupport.samplePrediction(flaggedForReview: true))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(flaggedForReview: true))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(flaggedForReview: true, clinicianOverride: "Uninfected"))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(flaggedForReview: false))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(flaggedForReview: false, clinicianOverride: "Parasitized"))

        let flagged = try fx.predictionRepo.flaggedForReview()
        XCTAssertEqual(flagged.count, 2)
        XCTAssertTrue(flagged.allSatisfy { $0.flaggedForReview && $0.clinicianOverride == nil })
    }

    // MARK: - Risk band

    func testRiskBand_low_grayZone_high() {
        // The shared GRAY_ZONE_LOW = 0.3, GRAY_ZONE_HIGH = 0.7 (spec §5).
        XCTAssertEqual(Double(Threshold.shared.GRAY_ZONE_LOW), 0.3, accuracy: 0.0001)
        XCTAssertEqual(Double(Threshold.shared.GRAY_ZONE_HIGH), 0.7, accuracy: 0.0001)

        XCTAssertEqual(RiskBandIndicator.band(for: 0.20), .low)
        XCTAssertEqual(RiskBandIndicator.band(for: 0.50), .grayZone)
        XCTAssertEqual(RiskBandIndicator.band(for: 0.90), .high)

        // Inclusive boundaries on the gray zone.
        XCTAssertEqual(RiskBandIndicator.band(for: Double(Threshold.shared.GRAY_ZONE_LOW)), .grayZone)
        XCTAssertEqual(RiskBandIndicator.band(for: Double(Threshold.shared.GRAY_ZONE_HIGH)), .grayZone)
    }

    // MARK: - Session-relabel ASCII validation

    func testRelabelValidation_acceptsPlainAscii() {
        XCTAssertTrue(SessionRelabelView.isValidLabel("morning slide"))
        XCTAssertTrue(SessionRelabelView.isValidLabel("A"))
        XCTAssertTrue(SessionRelabelView.isValidLabel("Slide 42 OK"))
        // Em-dash IS non-ASCII so this string must be rejected.
        XCTAssertFalse(SessionRelabelView.isValidLabel("Ward 3 — slide"))
    }

    func testRelabelValidation_rejectsEmojiAndNonAscii() {
        XCTAssertFalse(SessionRelabelView.isValidLabel("morning 🩸 slide"))
        XCTAssertFalse(SessionRelabelView.isValidLabel("café"))      // é is non-ASCII
        XCTAssertFalse(SessionRelabelView.isValidLabel("北京"))         // CJK
        XCTAssertFalse(SessionRelabelView.isValidLabel(""))            // empty
        XCTAssertFalse(SessionRelabelView.isValidLabel(String(repeating: "x", count: 21))) // > 20 chars
    }
}
