import SwiftData
import XCTest
@testable import MalariaDetector

@MainActor
final class SettingsTests: XCTestCase {

    private func makeStore(_ fx: TestSupport.Fixture) -> SettingsStore {
        SettingsStore(auditRepo: fx.auditRepo, audit: fx.auditLog, clinicians: fx.clinicianRepo)
    }

    private func makeOnboarding(_ fx: TestSupport.Fixture) -> OnboardingState {
        OnboardingState(
            clinicians: fx.clinicianRepo,
            consents: fx.consentRepo,
            audit: fx.auditLog
        )
    }

    func testHydrate_readsClinicNameFromAuditEntry() throws {
        let fx = try TestSupport.makeFixture()
        // Seed a clinic_configured entry — same metadata shape that
        // `OnboardingState.configureClinic(...)` writes during Phase 1.
        fx.auditLog.write(
            .clinicConfigured,
            actorId: "pre-provisioning",
            actorRoleAtTime: "admin",
            metadata: [
                "clinic_name": "Kisumu District Health Centre",
                "jurisdiction": "ke_dpa",
                "lawful_basis": "vital_interests",
            ]
        )

        let store = makeStore(fx)
        store.hydrate()

        XCTAssertEqual(store.clinicName, "Kisumu District Health Centre")
        XCTAssertEqual(store.jurisdiction, "ke_dpa")
        XCTAssertEqual(store.lawfulBasis, "vital_interests")
    }

    func testHydrate_readsInferencePolicyFromAuditEntry() throws {
        let fx = try TestSupport.makeFixture()
        fx.auditLog.write(
            .inferencePolicySet,
            actorId: "pre-provisioning",
            actorRoleAtTime: "admin",
            metadata: [
                "threshold": "0.42",
                "default_model": "EfficientNetB0_Keras",
                "auto_logout_minutes": "5",
            ]
        )

        let store = makeStore(fx)
        store.hydrate()

        XCTAssertEqual(store.threshold, 0.42, accuracy: 1e-9)
        XCTAssertEqual(store.defaultModelId, "EfficientNetB0_Keras")
        XCTAssertEqual(store.autoLogoutMinutes, 5)
    }

    func testUpdateThreshold_writesThresholdChangedAuditEntry() throws {
        let fx = try TestSupport.makeFixture()
        _ = try fx.clinicianRepo.enroll(role: "admin")
        fx.auditLog.write(
            .inferencePolicySet,
            actorId: "x",
            actorRoleAtTime: "admin",
            metadata: ["threshold": "0.3", "default_model": "BNLeaky_Keras", "auto_logout_minutes": "15"]
        )

        let store = makeStore(fx)
        store.hydrate()
        store.updateThreshold(0.55)

        let entries = try fx.auditRepo.entries(forAction: .thresholdChanged)
        XCTAssertEqual(entries.count, 1)
        let metadata = entries[0].metadataJson
        XCTAssertTrue(metadata.contains("\"old_value\""))
        XCTAssertTrue(metadata.contains("\"new_value\""))
        XCTAssertTrue(metadata.contains("0.55"))
        XCTAssertEqual(store.threshold, 0.55, accuracy: 1e-9)
    }

    func testUpdateDefaultModel_writesDefaultModelChangedAuditEntry() throws {
        let fx = try TestSupport.makeFixture()
        _ = try fx.clinicianRepo.enroll(role: "admin")

        let store = makeStore(fx)
        store.updateDefaultModel("MobileNetV3Large_Keras")

        let entries = try fx.auditRepo.entries(forAction: .defaultModelChanged)
        XCTAssertEqual(entries.count, 1)
        XCTAssertTrue(entries[0].metadataJson.contains("MobileNetV3Large_Keras"))
        XCTAssertEqual(store.defaultModelId, "MobileNetV3Large_Keras")
    }

    func testResetCoordinator_wipesClinicianButKeepsPredictionsAndAudit() throws {
        let fx = try TestSupport.makeFixture()
        // Seed: a clinician, a prediction, and a couple of audit entries.
        let admin = try fx.clinicianRepo.enroll(role: "admin")
        try fx.clinicianRepo.markBiometricEnrolled(admin)
        try fx.consentRepo.record(
            actorId: admin.actorId,
            consentType: .hippocraticLicense,
            documentVersion: "v3.0",
            value: "accepted"
        )
        _ = try fx.predictionRepo.insert(TestSupport.samplePrediction())
        fx.auditLog.write(.adminProvisioningStarted, actorId: admin.actorId, actorRoleAtTime: "admin")
        fx.auditLog.write(.adminProvisioningCompleted, actorId: admin.actorId, actorRoleAtTime: "admin")

        let store = makeStore(fx)
        store.hydrate()
        let onboarding = makeOnboarding(fx)
        let coordinator = ResetDeviceCoordinator(
            clinicians: fx.clinicianRepo,
            audit: fx.auditLog,
            onboarding: onboarding,
            settings: store
        )

        let predictionsBefore = try fx.predictionRepo.recent(limit: 100).count
        let auditCountBefore = try fx.auditRepo.count()

        try coordinator.performReset()

        // Clinician + consents... well, consents stay (spec §10 preserves
        // chain-of-custody on the audit log; the clinician row is the only
        // explicit wipe target). Predictions stay. Audit entries stay and
        // gain exactly one new `device_reprovisioned` row.
        XCTAssertNil(try fx.clinicianRepo.current())
        XCTAssertEqual(try fx.predictionRepo.recent(limit: 100).count, predictionsBefore)
        XCTAssertEqual(try fx.auditRepo.count(), auditCountBefore + 1)
        let reprovisioned = try fx.auditRepo.entries(forAction: .deviceReprovisioned)
        XCTAssertEqual(reprovisioned.count, 1)
        XCTAssertEqual(reprovisioned.first?.actorId, admin.actorId)
    }

    func testResetCoordinator_returnsOnboardingToAdminProvisioning() throws {
        let fx = try TestSupport.makeFixture()
        let admin = try fx.clinicianRepo.enroll(role: "admin")
        try fx.clinicianRepo.markBiometricEnrolled(admin)
        let onboarding = makeOnboarding(fx)
        // Simulate "complete" state — rehydrate sets `.complete` when
        // admin + biometric-enrolled.
        try onboarding.rehydrate()
        XCTAssertEqual(onboarding.phase, .complete)

        let store = makeStore(fx)
        let coordinator = ResetDeviceCoordinator(
            clinicians: fx.clinicianRepo,
            audit: fx.auditLog,
            onboarding: onboarding,
            settings: store
        )
        try coordinator.performReset()

        XCTAssertEqual(onboarding.phase, .adminProvisioning)
        XCTAssertEqual(onboarding.adminStep, .language)
        XCTAssertNil(onboarding.pendingClinicName)
    }
}
