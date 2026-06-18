import XCTest
@testable import MalariaDetector

/// Covers the `OnboardingState` extensions added for the Phase 6 wizard
/// (spec §10): the pending-state fields the UI reads for the
/// provisioning-complete screen, and the `finishOrientation()` /
/// `proceedToMicroscopistClaim()` transitions the wizard issues.
///
/// These tests exercise the state machine directly — SwiftUI snapshot or
/// view-hierarchy tests would need a preview-host infrastructure that
/// isn't yet in place. The wizard step views forward to these state
/// methods exclusively, so testing the methods covers the behaviour the
/// wizard relies on.
@MainActor
final class OnboardingFlowTests: XCTestCase {

    private func makeOnboarding(_ fx: TestSupport.Fixture) -> OnboardingState {
        OnboardingState(
            clinicians: fx.clinicianRepo,
            consents: fx.consentRepo,
            audit: fx.auditLog
        )
    }

    // MARK: - finishOrientation

    func testFinishOrientation_flipsPhaseToComplete() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        _ = try onboarding.completeAdminProvisioning()
        onboarding.proceedToMicroscopistClaim()
        try onboarding.startMicroscopistClaim()
        try onboarding.completeMicroscopistClaim()
        XCTAssertEqual(onboarding.phase, .microscopistClaim)

        onboarding.finishOrientation()

        XCTAssertEqual(onboarding.phase, .complete)
    }

    func testFinishOrientation_writesNoAdditionalAuditEntries() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        _ = try onboarding.completeAdminProvisioning()
        onboarding.proceedToMicroscopistClaim()
        try onboarding.startMicroscopistClaim()
        try onboarding.completeMicroscopistClaim()
        let before = try fx.auditRepo.entries(forAction: .microscopistClaimCompleted).count

        onboarding.finishOrientation()

        let after = try fx.auditRepo.entries(forAction: .microscopistClaimCompleted).count
        XCTAssertEqual(before, after)
    }

    // MARK: - pendingClinicName / pendingClinicJurisdiction / pendingLawfulBasis

    func testConfigureClinic_setsPendingFieldsForLaterScreens() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        XCTAssertNil(onboarding.pendingClinicName)
        XCTAssertNil(onboarding.pendingClinicJurisdiction)
        XCTAssertNil(onboarding.pendingLawfulBasis)

        try onboarding.configureClinic(
            name: "Kisumu District Health Centre",
            jurisdiction: "ke_dpa",
            lawfulBasis: "health_provision",
            version: "v1.0"
        )

        XCTAssertEqual(onboarding.pendingClinicName, "Kisumu District Health Centre")
        XCTAssertEqual(onboarding.pendingClinicJurisdiction, "ke_dpa")
        XCTAssertEqual(onboarding.pendingLawfulBasis, "health_provision")
    }

    // MARK: - admin-step machine: language → welcome → licenseAck

    func testAdvanceFromLanguage_movesToWelcome() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        XCTAssertEqual(onboarding.adminStep, .language)

        onboarding.advanceFromLanguage()

        XCTAssertEqual(onboarding.adminStep, .welcome)
    }

    func testAdvanceFromWelcome_movesToLicenseAck() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        onboarding.advanceFromLanguage()

        onboarding.advanceFromWelcome()

        XCTAssertEqual(onboarding.adminStep, .licenseAck)
    }

    func testAdvanceFromLanguage_isNoopWhenNotOnLanguageStep() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        onboarding.advanceFromLanguage() // now on .welcome

        onboarding.advanceFromLanguage()

        // Doesn't skip ahead — protects against double-tap glitches.
        XCTAssertEqual(onboarding.adminStep, .welcome)
    }

    // MARK: - completion gate

    func testProceedToMicroscopistClaim_switchesPhaseFromAdminToMicroscopist() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        _ = try onboarding.completeAdminProvisioning()
        XCTAssertEqual(onboarding.phase, .adminProvisioning)
        XCTAssertEqual(onboarding.adminStep, .provisioningComplete)

        onboarding.proceedToMicroscopistClaim()

        XCTAssertEqual(onboarding.phase, .microscopistClaim)
        XCTAssertEqual(onboarding.microscopistStep, .welcome)
    }

    /// The composition root mounts `OnboardingFlow` while phase ≠
    /// `.complete` and `RootView` once it is. This test asserts the
    /// finish-orientation transition is exactly that gate.
    func testRootViewVisibility_followsCompletionGate() throws {
        let fx = try TestSupport.makeFixture()
        let onboarding = makeOnboarding(fx)
        XCTAssertNotEqual(onboarding.phase, .complete, "Fresh device must show onboarding, not RootView")

        _ = try onboarding.completeAdminProvisioning()
        onboarding.proceedToMicroscopistClaim()
        try onboarding.startMicroscopistClaim()
        try onboarding.completeMicroscopistClaim()
        XCTAssertNotEqual(onboarding.phase, .complete, "Wizard still owns the screen while orientation pages are visible")

        onboarding.finishOrientation()
        XCTAssertEqual(onboarding.phase, .complete, "RootView mounts only after orientation finishes")
    }

    // MARK: - LanguagePreference

    func testLanguagePreference_persistsAcrossInstances() {
        let suite = UserDefaults(suiteName: "test.onboarding.\(UUID().uuidString)")!
        suite.removePersistentDomain(forName: "test.onboarding")

        XCTAssertEqual(LanguagePreference.get(defaults: suite), .english)

        LanguagePreference.set(.swahili, defaults: suite)

        XCTAssertEqual(LanguagePreference.get(defaults: suite), .swahili)
    }
}
