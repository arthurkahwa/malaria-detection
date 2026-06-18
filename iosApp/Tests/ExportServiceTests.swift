import Foundation
import SwiftData
import XCTest
@testable import MalariaDetector
@preconcurrency import Shared

/// Phase 13 export-bundle tests on the iOS side. Coverage focuses on the
/// composition: shared `ExportBundleBuilder` is exercised by commonTest, so
/// here we assert the file-system + audit-chain behaviour.
@MainActor
final class ExportServiceTests: XCTestCase {

    private func makeService(_ fx: TestSupport.Fixture) throws -> ExportService {
        let settings = SettingsStore(
            auditRepo: fx.auditRepo,
            audit: fx.auditLog,
            clinicians: fx.clinicianRepo
        )
        // Seed a clinic_configured row so the export has clinic context.
        fx.auditLog.write(
            .clinicConfigured,
            actorId: "admin",
            actorRoleAtTime: "admin",
            metadata: [
                "clinic_name": "Test Clinic",
                "jurisdiction": "ke_dpa",
                "lawful_basis": "vital_interests",
            ]
        )
        settings.hydrate()
        return ExportService(
            predictions: fx.predictionRepo,
            audits: fx.auditRepo,
            clinicians: fx.clinicianRepo,
            consents: fx.consentRepo,
            settings: settings,
            auditLog: fx.auditLog
        )
    }

    private func enrollClinician(_ fx: TestSupport.Fixture) throws -> ClinicianProfile {
        try fx.clinicianRepo.enroll(role: "microscopist", initials: "AB")
    }

    func testGenerateBundle_producesNonEmptyFileURL() async throws {
        let fx = try TestSupport.makeFixture()
        _ = try enrollClinician(fx)
        let service = try makeService(fx)

        let url = try await service.generateBundle()
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "bundle file should exist")
        let size = (try FileManager.default.attributesOfItem(atPath: url.path))[.size] as? Int ?? 0
        XCTAssertGreaterThan(size, 0, "bundle should not be empty")
    }

    func testGenerateBundle_writesAuditChain() async throws {
        let fx = try TestSupport.makeFixture()
        _ = try enrollClinician(fx)
        let service = try makeService(fx)

        let beforeInitiated = try fx.auditRepo.entries(forAction: .exportInitiated).count
        let beforeCompleted = try fx.auditRepo.entries(forAction: .exportCompleted).count
        _ = try await service.generateBundle()

        let afterInitiated = try fx.auditRepo.entries(forAction: .exportInitiated).count
        let afterCompleted = try fx.auditRepo.entries(forAction: .exportCompleted).count

        XCTAssertEqual(afterInitiated - beforeInitiated, 1, "expected exactly one export_initiated entry")
        XCTAssertEqual(afterCompleted - beforeCompleted, 1, "expected exactly one export_completed entry")

        // export_completed metadata carries size + signature per spec §14.
        let completed = try fx.auditRepo.entries(forAction: .exportCompleted).first
        XCTAssertNotNil(completed)
        let metadata = completed?.metadataJson ?? ""
        XCTAssertTrue(metadata.contains("\"signature\""), "expected signature in metadata, got: \(metadata)")
        XCTAssertTrue(metadata.contains("\"size\""), "expected size in metadata, got: \(metadata)")
    }

    func testGenerateBundle_filenameMatchesSpecPattern() async throws {
        let fx = try TestSupport.makeFixture()
        _ = try enrollClinician(fx)
        let service = try makeService(fx)

        let url = try await service.generateBundle()
        let name = url.lastPathComponent
        // Spec §14: `malaria-detector-export-{deviceUuid-prefix}-{timestamp}.zip`
        XCTAssertTrue(name.hasPrefix("malaria-detector-export-"), "got: \(name)")
        XCTAssertTrue(name.hasSuffix(".zip"), "got: \(name)")
        // Two hyphens separate the three parts after the literal prefix:
        // malaria-detector-export, {deviceUuid-prefix}, {timestamp}.zip
        XCTAssertGreaterThan(name.count, "malaria-detector-export-".count + 8, "too short: \(name)")
    }

    func testGenerateBundle_failsWithExportFailedAuditWhenClinicNotConfigured() async throws {
        let fx = try TestSupport.makeFixture()
        _ = try enrollClinician(fx)
        // Settings.hydrate() with no clinic_configured row leaves clinicName nil.
        let bareSettings = SettingsStore(
            auditRepo: fx.auditRepo,
            audit: fx.auditLog,
            clinicians: fx.clinicianRepo
        )
        bareSettings.hydrate()
        let service = ExportService(
            predictions: fx.predictionRepo,
            audits: fx.auditRepo,
            clinicians: fx.clinicianRepo,
            consents: fx.consentRepo,
            settings: bareSettings,
            auditLog: fx.auditLog
        )

        do {
            _ = try await service.generateBundle()
            XCTFail("expected ExportError.clinicConfigMissing")
        } catch {
            // Expected
        }

        XCTAssertEqual(
            try fx.auditRepo.entries(forAction: .exportFailed).count, 1,
            "expected exactly one export_failed entry"
        )
    }
}
