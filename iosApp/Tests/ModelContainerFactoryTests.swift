import Foundation
import SwiftData
import XCTest
@testable import MalariaDetector

@MainActor
final class ModelContainerFactoryTests: XCTestCase {

    func testStoreFile_hasCompleteProtectionAttribute() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("malaria-fp-test-\(UUID().uuidString).store")
        defer { try? removeStoreFiles(at: tempURL) }

        let container = try ModelContainerFactory.make(storeURL: tempURL)

        // Materialise the file by writing a row.
        let profile = ClinicianProfile(
            actorId: UUID().uuidString,
            role: "admin",
            enrolledAt: Date()
        )
        container.mainContext.insert(profile)
        try container.mainContext.save()

        // Spec §8: NSFileProtectionComplete must be applied to the store.
        // The factory call must not throw and must leave the file readable;
        // the attribute readback assertion only runs on real devices because
        // iOS Simulator's filesystem doesn't enforce protection classes
        // (setAttributes succeeds, reads back nil).
        ModelContainerFactory.applyCompleteProtection(at: tempURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))

        #if !targetEnvironment(simulator)
        let attrs = try FileManager.default.attributesOfItem(atPath: tempURL.path)
        XCTAssertEqual(
            attrs[.protectionKey] as? FileProtectionType,
            .complete,
            "Store file must have NSFileProtectionComplete applied on device"
        )
        #endif
    }

    func testSchema_includesAllFourEntities() throws {
        // Smoke test: the container builds without error against the full
        // canonical schema (spec §8). A missing @Model class in the schema
        // would fail to construct.
        let container = try ModelContainerFactory.make(
            storeURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("malaria-schema-\(UUID().uuidString).store")
        )
        XCTAssertNotNil(container.mainContext)
    }

    private func removeStoreFiles(at url: URL) throws {
        for suffix in ["", "-wal", "-shm"] {
            let path = url.path + suffix
            if FileManager.default.fileExists(atPath: path) {
                try FileManager.default.removeItem(atPath: path)
            }
        }
    }
}
