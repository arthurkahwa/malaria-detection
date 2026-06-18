import Foundation
import XCTest
@testable import MalariaDetector
@preconcurrency import Shared

/// Phase 14 / spec §16 — iOS-side crash log coverage. Shared module
/// `CrashLogRecordTest` / `RecentActionRingTest` cover the DTO + ring
/// behaviour; these tests focus on the file-system + audit-chain layer.
@MainActor
final class CrashLogTests: XCTestCase {

    private func makeTempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("crashlogs-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func sampleRecord(incidentId: String = UUID().uuidString) -> CrashLogRecord {
        CrashLogRecord(
            incidentId: incidentId,
            timestampIso8601: "2026-05-21T12:00:00Z",
            appVersion: "0.1.0-test",
            osVersion: "iOS Test",
            deviceModelClass: "iPhone15,2",
            stackTrace: "NSException\n\tat anywhere",
            recentActionTypes: ["prediction_created", "override_recorded"],
            memoryPressure: "resident_mb=42",
            deviceUnlocked: true
        )
    }

    private func makeStore(directory: URL, fx: TestSupport.Fixture) -> CrashLogStore {
        let authGate = AuthGate(audit: fx.auditLog, clinicians: fx.clinicianRepo)
        return CrashLogStore(
            directoryURL: directory,
            auditLog: fx.auditLog,
            authGate: authGate,
            clinicians: fx.clinicianRepo
        )
    }

    // MARK: - Round-trip

    func testWriteThenList_returnsTheRecord() throws {
        let fx = try TestSupport.makeFixture()
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let record = sampleRecord()
        let written = CrashLogWriter.writeRecord(record, into: dir)
        XCTAssertNotNil(written)

        let store = makeStore(directory: dir, fx: fx)
        XCTAssertEqual(store.entries.count, 1)
        XCTAssertEqual(store.entries.first?.incidentId, record.incidentId)

        let decoded = try store.read(store.entries[0])
        XCTAssertEqual(decoded.incidentId, record.incidentId)
        XCTAssertEqual(decoded.deviceModelClass, "iPhone15,2")
        XCTAssertEqual(decoded.recentActionTypes, ["prediction_created", "override_recorded"])
    }

    // MARK: - 30-day expiry

    func testInit_sweepsLogsOlderThanThirtyDays() throws {
        let fx = try TestSupport.makeFixture()
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Write three records; backdate the first one to 31 days ago.
        let recent = sampleRecord(incidentId: "00000000-0000-0000-0000-000000000001")
        let old = sampleRecord(incidentId: "00000000-0000-0000-0000-000000000002")
        _ = CrashLogWriter.writeRecord(recent, into: dir)
        let oldURL = CrashLogWriter.writeRecord(old, into: dir)
        XCTAssertNotNil(oldURL)

        // Stamp the old file 31 days ago.
        let thirtyOneDaysAgo = Date().addingTimeInterval(-31 * 24 * 60 * 60)
        try FileManager.default.setAttributes(
            [.modificationDate: thirtyOneDaysAgo],
            ofItemAtPath: oldURL!.path
        )

        let store = makeStore(directory: dir, fx: fx)
        XCTAssertEqual(store.entries.count, 1, "expected only the recent log to survive sweep")
        XCTAssertEqual(store.entries.first?.incidentId, recent.incidentId)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: oldURL!.path),
            "expected the 31-day-old log to be removed on init"
        )
    }

    // MARK: - Audit

    func testDidShare_writesCrashLogSharedWithIncidentUuid() throws {
        let fx = try TestSupport.makeFixture()
        _ = try fx.clinicianRepo.enroll(role: "microscopist", initials: "AB")
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let record = sampleRecord()
        _ = CrashLogWriter.writeRecord(record, into: dir)

        let store = makeStore(directory: dir, fx: fx)
        XCTAssertEqual(store.entries.count, 1)

        let before = try fx.auditRepo.entries(forAction: .crashLogShared).count
        store.didShare(store.entries[0])
        let after = try fx.auditRepo.entries(forAction: .crashLogShared).count
        XCTAssertEqual(after - before, 1, "expected exactly one crash_log_shared audit entry")

        let entry = try fx.auditRepo.entries(forAction: .crashLogShared).first
        XCTAssertEqual(entry?.resourceId, record.incidentId)
        // Spec §16 forbids metadata on the share entry; we passed `[:]`.
        XCTAssertEqual(entry?.metadataJson, "{}")
    }

    // MARK: - File protection (skip-on-simulator)

    func testWriteRecord_appliesCompleteFileProtectionOnDevice() throws {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let record = sampleRecord()
        let url = CrashLogWriter.writeRecord(record, into: dir)
        XCTAssertNotNil(url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url!.path))

        // Same skip-on-simulator pattern as Phase 4's
        // `ModelContainerFactoryTests.testStoreFile_hasCompleteProtectionAttribute`.
        // Simulator's filesystem doesn't enforce protection classes; we still
        // assert the write succeeded above.
        #if !targetEnvironment(simulator)
        let attrs = try FileManager.default.attributesOfItem(atPath: url!.path)
        XCTAssertEqual(
            attrs[.protectionKey] as? FileProtectionType,
            .complete,
            "Crash log file must have NSFileProtectionComplete applied on device"
        )
        #endif
    }

    // MARK: - Ring snapshot ↔ AuditLog plumbing

    func testAuditLogWrite_feedsTheRecentActionRing() throws {
        let fx = try TestSupport.makeFixture()
        _ = try fx.clinicianRepo.enroll(role: "microscopist", initials: "AB")
        // Clean ring so we don't see entries from other tests in this
        // process.
        RecentActionRing.companion.shared.clear()

        fx.auditLog.write(
            .predictionViewed,
            actorId: "actor",
            actorRoleAtTime: "microscopist"
        )
        fx.auditLog.write(
            .overrideRecorded,
            actorId: "actor",
            actorRoleAtTime: "microscopist"
        )
        let snapshot = RecentActionRing.companion.shared.snapshot()
        XCTAssertEqual(snapshot, ["prediction_viewed", "override_recorded"])
        RecentActionRing.companion.shared.clear()
    }
}
