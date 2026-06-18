import Foundation
import SwiftData
import XCTest
@testable import MalariaDetector

/// In-memory ModelContainer for unit tests so they're fast and isolated.
/// Identical schema to production — drift would be caught by the
/// schema-drift snapshot test, not by relying on test/prod parity here.
@MainActor
enum TestSupport {

    static func makeInMemoryContainer() throws -> ModelContainer {
        let schema = Schema([
            Prediction.self,
            AuditEntry.self,
            ClinicianProfile.self,
            ConsentRecord.self,
        ])
        let config = ModelConfiguration(
            schema: schema,
            isStoredInMemoryOnly: true,
            cloudKitDatabase: .none
        )
        return try ModelContainer(for: schema, configurations: [config])
    }

    /// Helper used by every test that needs a freshly-wired set of
    /// repositories pointing at an isolated in-memory store.
    struct Fixture {
        let container: ModelContainer
        let predictionRepo: PredictionRepository
        let auditRepo: AuditRepository
        let clinicianRepo: ClinicianRepository
        let consentRepo: ConsentRepository
        let auditLog: AuditLog
    }

    static func makeFixture() throws -> Fixture {
        let container = try makeInMemoryContainer()
        let context = container.mainContext
        let auditRepo = AuditRepository(context: context)
        return Fixture(
            container: container,
            predictionRepo: PredictionRepository(context: context),
            auditRepo: auditRepo,
            clinicianRepo: ClinicianRepository(context: context),
            consentRepo: ConsentRepository(context: context),
            auditLog: AuditLog(repository: auditRepo)
        )
    }

    /// Constructs a Prediction with sensible defaults so each test only
    /// names the fields that matter to it.
    static func samplePrediction(
        id: String = UUID().uuidString,
        sessionId: String = UUID().uuidString,
        timestamp: Date = Date(),
        parasitizedProb: Double = 0.85,
        label: String = "Parasitized",
        threshold: Double = 0.3,
        flaggedForReview: Bool = false,
        imageHash: String = String(repeating: "a", count: 64),
        modelId: String = "BNLeaky_Keras",
        clinicianOverride: String? = nil,
        duplicateOfId: String? = nil
    ) -> Prediction {
        Prediction(
            id: id,
            sessionId: sessionId,
            timestamp: timestamp,
            modelId: modelId,
            modelVersion: modelId,
            parasitizedProb: parasitizedProb,
            uninfectedProb: 1.0 - parasitizedProb,
            label: label,
            threshold: threshold,
            flaggedForReview: flaggedForReview,
            inferenceMs: 42,
            imageHash: imageHash,
            clinicianOverride: clinicianOverride,
            duplicateOfId: duplicateOfId,
            appVersion: "0.1.0-test",
            osVersion: "iOS Test"
        )
    }
}
