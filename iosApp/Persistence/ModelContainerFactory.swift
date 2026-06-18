import Foundation
import SwiftData

/// Constructs the app's `ModelContainer` per spec §19 ("iOS app —
/// SwiftData container setup"):
/// - Schema covers all four persisted entities
/// - Store file lives in Application Support
/// - File protection class set to `.complete` so the store is
///   cryptographically unreadable when the device is locked
/// - `cloudKitDatabase: .none` — medical data MUST NOT sync to iCloud
enum ModelContainerFactory {

    /// Default location: `<App Support>/Malaria.store`.
    static func defaultStoreURL() throws -> URL {
        try FileManager.default
            .url(for: .applicationSupportDirectory, in: .userDomainMask,
                 appropriateFor: nil, create: true)
            .appendingPathComponent("Malaria.store")
    }

    /// Production builder. Use `make(storeURL:)` with a temp path in tests.
    @MainActor
    static func make() throws -> ModelContainer {
        try make(storeURL: try defaultStoreURL())
    }

    /// Test-friendly builder accepting an explicit store URL.
    @MainActor
    static func make(storeURL: URL) throws -> ModelContainer {
        let schema = Schema([
            Prediction.self,
            AuditEntry.self,
            ClinicianProfile.self,
            ConsentRecord.self,
        ])

        let config = ModelConfiguration(
            schema: schema,
            url: storeURL,
            cloudKitDatabase: .none
        )

        let container = try ModelContainer(for: schema, configurations: [config])

        // Apply file protection AFTER the container creates the file. Done
        // best-effort: if SwiftData hasn't materialised the file yet (first
        // launch before any insert) the setAttributes call is a no-op and we
        // re-apply on next insert via the repository layer.
        applyCompleteProtection(at: storeURL)

        return container
    }

    /// Applies `NSFileProtectionComplete` to the store file (and the
    /// SwiftData sidecar files that SQLite creates: `-wal`, `-shm`).
    /// Spec §8: HIPAA §164.312(a)(2)(iv) encryption-at-rest mitigation.
    static func applyCompleteProtection(at storeURL: URL) {
        let candidates = [
            storeURL.path,
            storeURL.path + "-wal",
            storeURL.path + "-shm",
        ]
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            try? FileManager.default.setAttributes(
                [.protectionKey: FileProtectionType.complete],
                ofItemAtPath: path
            )
        }
    }
}
