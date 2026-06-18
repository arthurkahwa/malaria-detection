import XCTest
@testable import MalariaDetector

@MainActor
final class OnboardingStateTests: XCTestCase {

    private func makeOnboarding(_ fx: TestSupport.Fixture) -> OnboardingState {
        OnboardingState(
            clinicians: fx.clinicianRepo,
            consents: fx.consentRepo,
            audit: fx.auditLog
        )
    }

    func testFreshDevice_startsAtAdminProvisioning() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)

        XCTAssertEqual(onboarding.phase, .adminProvisioning)
        XCTAssertEqual(onboarding.adminStep, .language)
    }

    func testAcceptHippocraticLicense_writesConsentAndAdvances() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)

        try onboarding.acceptHippocraticLicense(version: "v3.0")

        let records = try fx.consentRepo.records(for: "pre-provisioning")
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records.first?.consentType, "hippocratic_license")
        XCTAssertEqual(records.first?.documentVersion, "v3.0")
        XCTAssertEqual(records.first?.value, "accepted")
        XCTAssertEqual(onboarding.adminStep, .disclaimerAck)
    }

    func testCompleteAdminProvisioning_createsAdminProfileAndAdvancesToCompletionInterstitial() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)

        let profile = try onboarding.completeAdminProvisioning()

        XCTAssertEqual(profile.role, "admin")
        XCTAssertTrue(profile.biometricEnrolled)
        // Per spec §10 step 8: the admin sees a "Device provisioned for
        // [Clinic name]" interstitial before the microscopist phase
        // begins. Phase stays `.adminProvisioning` until the admin taps
        // through that screen via `proceedToMicroscopistClaim()`.
        XCTAssertEqual(onboarding.phase, .adminProvisioning)
        XCTAssertEqual(onboarding.adminStep, .provisioningComplete)

        // Two audit entries: biometric enrolment + provisioning completed.
        let completed = try fx.auditRepo.entries(forAction: .adminProvisioningCompleted)
        let biometric = try fx.auditRepo.entries(forAction: .adminBiometricEnrolled)
        XCTAssertEqual(completed.count, 1)
        XCTAssertEqual(biometric.count, 1)
        XCTAssertEqual(completed.first?.actorId, profile.actorId)
    }

    func testCompleteMicroscopistClaim_advancesToOrientation() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        _ = try onboarding.completeAdminProvisioning()
        onboarding.proceedToMicroscopistClaim()
        try onboarding.startMicroscopistClaim()

        try onboarding.completeMicroscopistClaim()

        // Per spec §10 Phase 2: orientation precedes "Begin screening".
        // The wizard stays in microscopistClaim phase until the user
        // dismisses or finishes orientation via `finishOrientation()`.
        XCTAssertEqual(onboarding.phase, .microscopistClaim)
        XCTAssertEqual(onboarding.microscopistStep, .orientation)
        let entries = try fx.auditRepo.entries(forAction: .microscopistClaimCompleted)
        XCTAssertEqual(entries.count, 1)
    }

    func testConfigureClinic_recordsJurisdictionAndLawfulBasisConsents() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)

        try onboarding.configureClinic(
            name: "Test Clinic",
            jurisdiction: "ke_dpa",
            lawfulBasis: "vital_interests",
            version: "v1"
        )

        let records = try fx.consentRepo.records(for: "pre-provisioning")
        XCTAssertEqual(records.count, 2)
        XCTAssertTrue(records.contains { $0.consentType == "jurisdiction" && $0.value == "ke_dpa" })
        XCTAssertTrue(records.contains { $0.consentType == "lawful_basis" && $0.value == "vital_interests" })

        let auditEntries = try fx.auditRepo.entries(forAction: .clinicConfigured)
        XCTAssertEqual(auditEntries.count, 1)
        XCTAssertEqual(auditEntries.first?.metadataJson.contains("Test Clinic"), true)
    }

    func testRehydrate_provisionedDevice_jumpsToComplete() throws {
        let fx = try TestSupport.makeFixture()
        let first = makeOnboarding(fx)
        _ = try first.completeAdminProvisioning()
        first.proceedToMicroscopistClaim()
        try first.startMicroscopistClaim()
        try first.completeMicroscopistClaim()
        first.finishOrientation()

        // Simulate app relaunch: new OnboardingState reads persisted state.
        let second = makeOnboarding(fx)
        XCTAssertEqual(second.phase, .complete)
    }
}
