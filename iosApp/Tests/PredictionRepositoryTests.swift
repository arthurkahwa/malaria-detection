import SwiftData
import XCTest
@testable import MalariaDetector

@MainActor
final class PredictionRepositoryTests: XCTestCase {

    func testInsert_appendsToStore() throws {
        let fx = try TestSupport.makeFixture()
        let p = TestSupport.samplePrediction()

        try fx.predictionRepo.insert(p)

        let all = try fx.predictionRepo.recent()
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all.first?.id, p.id)
    }

    func testRecent_returnsNewestFirst() throws {
        let fx = try TestSupport.makeFixture()
        let now = Date()
        try fx.predictionRepo.insert(TestSupport.samplePrediction(timestamp: now.addingTimeInterval(-60)))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(timestamp: now))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(timestamp: now.addingTimeInterval(-30)))

        let recent = try fx.predictionRepo.recent()

        XCTAssertEqual(recent.count, 3)
        XCTAssertEqual(recent[0].timestamp, now)
    }

    func testInSession_returnsOnlyMatchingSession() throws {
        let fx = try TestSupport.makeFixture()
        let sessionA = UUID().uuidString
        let sessionB = UUID().uuidString
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionA))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionA))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionB))

        let inA = try fx.predictionRepo.inSession(sessionA)
        XCTAssertEqual(inA.count, 2)
        XCTAssertTrue(inA.allSatisfy { $0.sessionId == sessionA })
    }

    func testFlaggedForReview_excludesAlreadyOverridden() throws {
        let fx = try TestSupport.makeFixture()
        try fx.predictionRepo.insert(TestSupport.samplePrediction(flaggedForReview: true))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(flaggedForReview: true, clinicianOverride: "Uninfected"))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(flaggedForReview: false))

        let flagged = try fx.predictionRepo.flaggedForReview()
        XCTAssertEqual(flagged.count, 1)
    }

    func testRecordOverride_setsFields() throws {
        let fx = try TestSupport.makeFixture()
        let p = TestSupport.samplePrediction(flaggedForReview: true)
        try fx.predictionRepo.insert(p)

        try fx.predictionRepo.recordOverride(on: p, verdict: "Uninfected", context: "review")

        let fetched = try fx.predictionRepo.byId(p.id)
        XCTAssertEqual(fetched?.clinicianOverride, "Uninfected")
        XCTAssertEqual(fetched?.overrideContext, "review")
    }

    func testMarkAsDuplicate_setsDuplicateOfId() throws {
        let fx = try TestSupport.makeFixture()
        let original = TestSupport.samplePrediction()
        let duplicate = TestSupport.samplePrediction()
        try fx.predictionRepo.insert(original)
        try fx.predictionRepo.insert(duplicate)

        try fx.predictionRepo.markAsDuplicate(duplicate, of: original.id)

        let fetched = try fx.predictionRepo.byId(duplicate.id)
        XCTAssertEqual(fetched?.duplicateOfId, original.id)
    }

    func testRelabel_appliesLabelToAllInSession() throws {
        let fx = try TestSupport.makeFixture()
        let sessionId = UUID().uuidString
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionId))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(sessionId: sessionId))

        try fx.predictionRepo.relabel(sessionId: sessionId, label: "morning slide")

        let inSession = try fx.predictionRepo.inSession(sessionId)
        XCTAssertEqual(inSession.count, 2)
        XCTAssertTrue(inSession.allSatisfy { $0.sessionLabel == "morning slide" })
    }

    func testMostRecent_returnsLatestTimestamp() throws {
        let fx = try TestSupport.makeFixture()
        let now = Date()
        try fx.predictionRepo.insert(TestSupport.samplePrediction(timestamp: now.addingTimeInterval(-60)))
        try fx.predictionRepo.insert(TestSupport.samplePrediction(timestamp: now))

        let mostRecent = try fx.predictionRepo.mostRecent()
        XCTAssertEqual(mostRecent?.timestamp, now)
    }
}
