import SwiftData
import XCTest
@testable import MalariaDetector

@MainActor
final class AuditRepositoryTests: XCTestCase {

    func testWrite_assignsMonotonicSeq() throws {
        let fx = try TestSupport.makeFixture()

        let a = try fx.auditRepo.write(action: .sessionUnlocked, actorId: "u", actorRoleAtTime: "admin")
        let b = try fx.auditRepo.write(action: .sessionUnlocked, actorId: "u", actorRoleAtTime: "admin")
        let c = try fx.auditRepo.write(action: .sessionUnlocked, actorId: "u", actorRoleAtTime: "admin")

        XCTAssertEqual(a.seq, 1)
        XCTAssertEqual(b.seq, 2)
        XCTAssertEqual(c.seq, 3)
    }

    func testWrite_storesCanonicalActionString() throws {
        let fx = try TestSupport.makeFixture()

        let entry = try fx.auditRepo.write(
            action: .predictionCreated,
            actorId: "u",
            actorRoleAtTime: "microscopist"
        )

        // Canonical English regardless of UI locale (spec §5/§8).
        XCTAssertEqual(entry.action, "prediction_created")
    }

    func testWrite_encodesMetadataAsSortedJson() throws {
        let fx = try TestSupport.makeFixture()

        let entry = try fx.auditRepo.write(
            action: .overrideRecorded,
            actorId: "u",
            actorRoleAtTime: "admin",
            metadata: ["model_id": "BNLeaky_Keras", "label": "Parasitized"]
        )

        // Sorted-keys output is deterministic — important so export bundles
        // are byte-stable across exports of the same content.
        XCTAssertEqual(entry.metadataJson, "{\"label\":\"Parasitized\",\"model_id\":\"BNLeaky_Keras\"}")
    }

    func testWrite_emptyMetadata_writesEmptyObject() throws {
        let fx = try TestSupport.makeFixture()

        let entry = try fx.auditRepo.write(
            action: .sessionUnlocked,
            actorId: "u",
            actorRoleAtTime: "admin"
        )

        XCTAssertEqual(entry.metadataJson, "{}")
    }

    func testWrite_overrideFields_persistOnEntry() throws {
        let fx = try TestSupport.makeFixture()

        let entry = try fx.auditRepo.write(
            action: .overrideRecorded,
            actorId: "u",
            actorRoleAtTime: "microscopist",
            overrideContext: "review",
            overrideReason: "image_quality",
            overrideNotes: "blurry",
            contextReviewed: true,
            overrideActorInitials: "JM"
        )

        XCTAssertEqual(entry.overrideContext, "review")
        XCTAssertEqual(entry.overrideReason, "image_quality")
        XCTAssertEqual(entry.overrideNotes, "blurry")
        XCTAssertEqual(entry.contextReviewed, true)
        XCTAssertEqual(entry.overrideActorInitials, "JM")
    }

    func testRecent_returnsHighestSeqFirst() throws {
        let fx = try TestSupport.makeFixture()
        for _ in 0..<5 {
            _ = try fx.auditRepo.write(action: .sessionUnlocked, actorId: "u", actorRoleAtTime: "admin")
        }

        let recent = try fx.auditRepo.recent(limit: 3)

        XCTAssertEqual(recent.count, 3)
        XCTAssertEqual(recent.map(\.seq), [5, 4, 3])
    }

    func testEntriesForAction_filtersByCanonicalString() throws {
        let fx = try TestSupport.makeFixture()
        _ = try fx.auditRepo.write(action: .sessionUnlocked, actorId: "u", actorRoleAtTime: "admin")
        _ = try fx.auditRepo.write(action: .predictionCreated, actorId: "u", actorRoleAtTime: "admin")
        _ = try fx.auditRepo.write(action: .sessionUnlocked, actorId: "u", actorRoleAtTime: "admin")

        let unlocks = try fx.auditRepo.entries(forAction: .sessionUnlocked)
        XCTAssertEqual(unlocks.count, 2)
    }

    func testCount_matchesInsertedRows() throws {
        let fx = try TestSupport.makeFixture()
        for _ in 0..<7 {
            _ = try fx.auditRepo.write(action: .sessionUnlocked, actorId: "u", actorRoleAtTime: "admin")
        }

        XCTAssertEqual(try fx.auditRepo.count(), 7)
    }
}
