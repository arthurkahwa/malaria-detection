import Foundation
import Observation

/// Drives the two-phase onboarding flow (spec §10):
///
/// Phase 1 (admin provisioning) → Phase 2 (microscopist claim) → operational.
///
/// Each step persists state through the repositories and writes the
/// matching audit entry. The UI observes `phase` and `adminStep` /
/// `microscopistStep` to decide which wizard screen to render — that
/// onboarding-wizard UI itself is a Phase 6/7 deliverable; this service
/// provides the state machine and persistence wiring it'll consume.
@Observable
@MainActor
final class OnboardingState {

    enum Phase: Sendable, Equatable {
        case adminProvisioning
        case microscopistClaim
        case complete
    }

    enum AdminStep: Sendable, Equatable {
        case language
        case welcome
        case licenseAck
        case disclaimerAck
        case clinicDetails
        case inferencePolicy
        case biometric
        /// Spec §10 step 8 — "Device provisioned for [Clinic name]. Hand
        /// to microscopist to complete setup." A dedicated step (rather
        /// than skipping straight to `.microscopistClaim`) so the
        /// single-person-deployment offer can render cleanly.
        case provisioningComplete
    }

    enum MicroscopistStep: Sendable, Equatable {
        case welcome
        case initials
        case biometric
        case orientation
    }

    private(set) var phase: Phase = .adminProvisioning
    private(set) var adminStep: AdminStep = .language
    private(set) var microscopistStep: MicroscopistStep = .welcome

    /// In-flight Phase 1 clinic-level state. The wizard writes these as
    /// each step completes so later steps (e.g. the "Provisioning complete"
    /// screen displaying clinic name) and the post-completion microscopist
    /// welcome screen can read them. v1 doesn't persist these as their own
    /// rows — they live in the `clinic_configured` audit entry's metadata,
    /// which is the chain-of-custody source of truth. Holding them here
    /// avoids re-parsing audit JSON on every UI render.
    private(set) var pendingClinicName: String?
    private(set) var pendingClinicJurisdiction: String?
    private(set) var pendingLawfulBasis: String?

    private let clinicians: ClinicianRepository
    private let consents: ConsentRepository
    private let audit: AuditLog

    init(
        clinicians: ClinicianRepository,
        consents: ConsentRepository,
        audit: AuditLog
    ) {
        self.clinicians = clinicians
        self.consents = consents
        self.audit = audit
        try? rehydrate()
    }

    /// Inspects the persisted state on launch and resumes the wizard at
    /// the correct phase. A fully-provisioned device returns straight to
    /// `.complete`.
    func rehydrate() throws {
        guard let admin = try clinicians.current() else {
            phase = .adminProvisioning
            return
        }
        if admin.role == "admin" && admin.biometricEnrolled {
            // Admin provisioning is done; check whether microscopist
            // claim has completed. v1 single-clinician model: the admin
            // profile *becomes* the microscopist on completion in the
            // single-person deployment path.
            phase = .complete
        } else {
            phase = .adminProvisioning
            adminStep = .biometric
        }
    }

    // MARK: - Phase 1 advancement

    func startAdminProvisioning() {
        phase = .adminProvisioning
        adminStep = .language
        audit.write(.adminProvisioningStarted, actorId: "pre-provisioning", actorRoleAtTime: "admin")
    }

    /// Advance from the language picker to the welcome step. The language
    /// itself is persisted by the wizard view (UserDefaults) — this method
    /// just walks the step machine.
    func advanceFromLanguage() {
        guard adminStep == .language else { return }
        adminStep = .welcome
    }

    /// Advance from the welcome blurb to the Hippocratic license screen.
    func advanceFromWelcome() {
        guard adminStep == .welcome else { return }
        adminStep = .licenseAck
    }

    func acceptHippocraticLicense(version: String) throws {
        try recordConsent(.hippocraticLicense, version: version, value: "accepted")
        adminStep = .disclaimerAck
    }

    func acceptMedicalDisclaimer(version: String) throws {
        try recordConsent(.medicalDisclaimer, version: version, value: "accepted")
        adminStep = .clinicDetails
    }

    func configureClinic(
        name: String,
        jurisdiction: String,
        lawfulBasis: String,
        version: String
    ) throws {
        try recordConsent(.jurisdiction, version: version, value: jurisdiction)
        try recordConsent(.lawfulBasis, version: version, value: lawfulBasis)
        audit.write(
            .clinicConfigured,
            actorId: pendingActorId(),
            actorRoleAtTime: "admin",
            metadata: ["clinic_name": name, "jurisdiction": jurisdiction, "lawful_basis": lawfulBasis]
        )
        pendingClinicName = name
        pendingClinicJurisdiction = jurisdiction
        pendingLawfulBasis = lawfulBasis
        adminStep = .inferencePolicy
    }

    func setInferencePolicy(threshold: Double, defaultModelId: String, autoLogoutMinutes: Int) {
        audit.write(
            .inferencePolicySet,
            actorId: pendingActorId(),
            actorRoleAtTime: "admin",
            metadata: [
                "threshold": String(threshold),
                "default_model": defaultModelId,
                "auto_logout_minutes": String(autoLogoutMinutes),
            ]
        )
        adminStep = .biometric
    }

    /// Enrol the admin profile + biometric. After this Phase 1 is done
    /// and the device is `provisioned-unclaimed` per spec §10.
    @discardableResult
    func completeAdminProvisioning() throws -> ClinicianProfile {
        let profile = try clinicians.enroll(role: "admin")
        try clinicians.markBiometricEnrolled(profile)
        audit.write(
            .adminBiometricEnrolled,
            actorId: profile.actorId,
            actorRoleAtTime: "admin",
            resourceType: "clinician",
            resourceId: profile.actorId
        )
        audit.write(
            .adminProvisioningCompleted,
            actorId: profile.actorId,
            actorRoleAtTime: "admin",
            resourceType: "clinician",
            resourceId: profile.actorId
        )
        // The admin wizard renders one more interstitial ("provisioned
        // for [Clinic name]") before swapping to the microscopist phase.
        adminStep = .provisioningComplete
        return profile
    }

    /// Called by the admin "Provisioning complete" screen after the
    /// admin either taps "Done" (hand device to microscopist later) or
    /// "Continue Phase 2 now" (single-person deployment path).
    func proceedToMicroscopistClaim() {
        phase = .microscopistClaim
        microscopistStep = .welcome
    }

    // MARK: - Phase 2 advancement

    func startMicroscopistClaim() throws {
        guard let admin = try clinicians.current() else {
            throw OnboardingError.adminNotProvisioned
        }
        audit.write(
            .microscopistClaimStarted,
            actorId: admin.actorId,
            actorRoleAtTime: admin.role
        )
        microscopistStep = .initials
    }

    func setMicroscopistInitials(_ initials: String?) throws {
        guard let profile = try clinicians.current() else {
            throw OnboardingError.adminNotProvisioned
        }
        try clinicians.updateInitials(initials, on: profile)
        microscopistStep = .biometric
    }

    /// Microscopist biometric enrolment. In single-person deployments the
    /// admin and microscopist are the same human; we keep two audit
    /// entries to preserve the chain-of-custody story.
    ///
    /// After this returns the wizard advances to the (skippable) orientation
    /// pages; `finishOrientation()` flips `phase` to `.complete` once the
    /// user dismisses or completes them. Phase stays `.microscopistClaim`
    /// during orientation so the wizard coordinator continues rendering
    /// the in-flight UI instead of swapping to `RootView`.
    func completeMicroscopistClaim() throws {
        guard let profile = try clinicians.current() else {
            throw OnboardingError.adminNotProvisioned
        }
        audit.write(
            .microscopistBiometricEnrolled,
            actorId: profile.actorId,
            actorRoleAtTime: profile.role,
            resourceType: "clinician",
            resourceId: profile.actorId
        )
        audit.write(
            .microscopistClaimCompleted,
            actorId: profile.actorId,
            actorRoleAtTime: profile.role,
            resourceType: "clinician",
            resourceId: profile.actorId
        )
        microscopistStep = .orientation
    }

    /// Final transition out of the wizard. Called when the microscopist
    /// either finishes or skips the three orientation pages. The audit
    /// trail for the claim is already written by `completeMicroscopistClaim`;
    /// no additional entry is needed for orientation per spec §10.
    func finishOrientation() {
        phase = .complete
    }

    /// Reset the in-memory state machine after a `Reset device` operation
    /// (spec §10 re-onboarding). The composition root reads `phase` to
    /// decide whether to show `OnboardingFlow` vs `RootView`, so flipping
    /// it back to `.adminProvisioning` synchronously is what wires the
    /// auto-relaunch into the admin wizard.
    ///
    /// Clinic-level pending fields are cleared too — they belong to the
    /// previous provisioning context and would mislead the wizard if a
    /// future step read them mid-flow. The canonical clinic record stays
    /// on the audit log (`clinic_configured`) per spec §10's chain-of-
    /// custody guarantee.
    func reset() {
        phase = .adminProvisioning
        adminStep = .language
        microscopistStep = .welcome
        pendingClinicName = nil
        pendingClinicJurisdiction = nil
        pendingLawfulBasis = nil
    }

    // MARK: - Testing bypass

#if DEBUG
    /// Forces `phase` to `.complete` without touching the database. Called
    /// from the app root when the `--skip-onboarding` launch argument is
    /// present so UI tests and screenshot automation land directly in
    /// `RootView` on a clean (empty) store.
    func skipForTesting() {
        phase = .complete
    }
#endif

    // MARK: - Private

    private func recordConsent(_ type: ConsentType, version: String, value: String) throws {
        try consents.record(
            actorId: pendingActorId(),
            consentType: type,
            documentVersion: version,
            value: value
        )
    }

    /// During Phase 1, the admin actor doesn't exist as a row yet. We use
    /// a placeholder identifier on the in-flight consent records; once
    /// `completeAdminProvisioning()` runs, follow-up records have a real
    /// `actorId`. Pre-provisioning consent rows can be re-stamped at the
    /// end of Phase 1 in a future migration if a deployer wants strict FK
    /// integrity; v1 accepts the placeholder as a known limitation.
    private func pendingActorId() -> String {
        (try? clinicians.current())?.actorId ?? "pre-provisioning"
    }
}

enum OnboardingError: LocalizedError {
    case adminNotProvisioned

    var errorDescription: String? {
        switch self {
        case .adminNotProvisioned: "Admin provisioning has not completed."
        }
    }
}
