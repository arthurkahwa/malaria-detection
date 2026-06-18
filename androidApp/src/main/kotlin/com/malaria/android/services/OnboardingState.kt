package com.malaria.android.services

import com.malaria.android.data.BuildEnvironment
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.dao.ConsentDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.data.entities.ConsentType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Drives the two-phase onboarding flow (spec §10):
 *
 * Phase 1 (admin provisioning) → Phase 2 (microscopist claim) → operational.
 *
 * Mirrors `iosApp/Services/OnboardingState.swift`. Each step persists via the
 * DAOs and writes the matching audit entry. The UI observes `phase`,
 * `adminStep`, and `microscopistStep` to decide which wizard screen to
 * render.
 */
class OnboardingState(
    private val clinicians: ClinicianDao,
    private val consents: ConsentDao,
    private val audit: AuditLog,
) {

    enum class Phase { AdminProvisioning, MicroscopistClaim, Complete }

    enum class AdminStep {
        Language,
        Welcome,
        LicenseAck,
        DisclaimerAck,
        ClinicDetails,
        InferencePolicy,
        Biometric,

        /**
         * Spec §10 step 8 — "Device provisioned for [Clinic name]. Hand to
         * microscopist to complete setup." A dedicated step (rather than
         * skipping straight to MicroscopistClaim) so the single-person
         * deployment offer can render cleanly.
         */
        ProvisioningComplete,
    }

    enum class MicroscopistStep { Welcome, Initials, Biometric, Orientation }

    private val _phase = MutableStateFlow(Phase.AdminProvisioning)
    val phase: StateFlow<Phase> = _phase.asStateFlow()

    private val _adminStep = MutableStateFlow(AdminStep.Language)
    val adminStep: StateFlow<AdminStep> = _adminStep.asStateFlow()

    private val _microscopistStep = MutableStateFlow(MicroscopistStep.Welcome)
    val microscopistStep: StateFlow<MicroscopistStep> = _microscopistStep.asStateFlow()

    /**
     * In-flight Phase 1 clinic-level state. The wizard writes these as each
     * step completes so later steps (e.g. the "Provisioning complete" screen
     * displaying clinic name) and the post-completion microscopist welcome
     * screen can read them. v1 doesn't persist these as their own rows —
     * they live in the `clinic_configured` audit entry's metadata, which is
     * the chain-of-custody source of truth. Holding them here avoids
     * re-parsing audit JSON on every UI render.
     */
    private val _pendingClinicName = MutableStateFlow<String?>(null)
    val pendingClinicName: StateFlow<String?> = _pendingClinicName.asStateFlow()

    private val _pendingClinicJurisdiction = MutableStateFlow<String?>(null)
    val pendingClinicJurisdiction: StateFlow<String?> = _pendingClinicJurisdiction.asStateFlow()

    private val _pendingLawfulBasis = MutableStateFlow<String?>(null)
    val pendingLawfulBasis: StateFlow<String?> = _pendingLawfulBasis.asStateFlow()

    /**
     * Inspect persisted state and resume the wizard at the right phase. A
     * fully-provisioned device returns straight to [Phase.Complete].
     * Mirrors iOS `OnboardingState.rehydrate()`.
     */
    suspend fun rehydrate() {
        val admin = runCatching { clinicians.current() }.getOrNull()
        if (admin == null) {
            _phase.value = Phase.AdminProvisioning
            return
        }
        if (admin.role == "admin" && admin.biometricEnrolled) {
            _phase.value = Phase.Complete
        } else {
            _phase.value = Phase.AdminProvisioning
            _adminStep.value = AdminStep.Biometric
        }
    }

    // -- Phase 1 advancement ----------------------------------------------

    suspend fun startAdminProvisioning() {
        _phase.value = Phase.AdminProvisioning
        _adminStep.value = AdminStep.Language
        audit.write(
            action = AuditAction.AdminProvisioningStarted,
            actorId = "pre-provisioning",
            actorRoleAtTime = "admin",
        )
    }

    /**
     * Advance from the language picker to the welcome step. The language
     * itself is persisted by the wizard view (DataStore) — this method just
     * walks the step machine.
     */
    fun advanceFromLanguage() {
        if (_adminStep.value != AdminStep.Language) return
        _adminStep.value = AdminStep.Welcome
    }

    /** Advance from the welcome blurb to the Hippocratic license screen. */
    fun advanceFromWelcome() {
        if (_adminStep.value != AdminStep.Welcome) return
        _adminStep.value = AdminStep.LicenseAck
    }

    suspend fun acceptHippocraticLicense(version: String) {
        recordConsent(ConsentType.HippocraticLicense, version, "accepted")
        _adminStep.value = AdminStep.DisclaimerAck
    }

    suspend fun acceptMedicalDisclaimer(version: String) {
        recordConsent(ConsentType.MedicalDisclaimer, version, "accepted")
        _adminStep.value = AdminStep.ClinicDetails
    }

    suspend fun configureClinic(
        name: String,
        jurisdiction: String,
        lawfulBasis: String,
        version: String,
    ) {
        recordConsent(ConsentType.Jurisdiction, version, jurisdiction)
        recordConsent(ConsentType.LawfulBasis, version, lawfulBasis)
        audit.write(
            action = AuditAction.ClinicConfigured,
            actorId = pendingActorId(),
            actorRoleAtTime = "admin",
            metadata = mapOf(
                "clinic_name" to name,
                "jurisdiction" to jurisdiction,
                "lawful_basis" to lawfulBasis,
            ),
        )
        _pendingClinicName.value = name
        _pendingClinicJurisdiction.value = jurisdiction
        _pendingLawfulBasis.value = lawfulBasis
        _adminStep.value = AdminStep.InferencePolicy
    }

    suspend fun setInferencePolicy(
        threshold: Double,
        defaultModelId: String,
        autoLogoutMinutes: Int,
    ) {
        audit.write(
            action = AuditAction.InferencePolicySet,
            actorId = pendingActorId(),
            actorRoleAtTime = "admin",
            metadata = mapOf(
                "threshold" to threshold.toString(),
                "default_model" to defaultModelId,
                "auto_logout_minutes" to autoLogoutMinutes.toString(),
            ),
        )
        _adminStep.value = AdminStep.Biometric
    }

    /**
     * Enrol the admin profile + biometric. After this, Phase 1 is done and
     * the device is `provisioned-unclaimed` per spec §10. The admin wizard
     * renders one more interstitial ([AdminStep.ProvisioningComplete])
     * before [proceedToMicroscopistClaim] flips the phase.
     */
    suspend fun completeAdminProvisioning(): ClinicianProfile {
        val profile = clinicians.enroll(role = "admin")
        clinicians.markBiometricEnrolled(profile.id)
        audit.write(
            action = AuditAction.AdminBiometricEnrolled,
            actorId = profile.actorId,
            actorRoleAtTime = "admin",
            resourceType = "clinician",
            resourceId = profile.actorId,
        )
        audit.write(
            action = AuditAction.AdminProvisioningCompleted,
            actorId = profile.actorId,
            actorRoleAtTime = "admin",
            resourceType = "clinician",
            resourceId = profile.actorId,
        )
        _adminStep.value = AdminStep.ProvisioningComplete
        return profile
    }

    /**
     * Called by the admin "Provisioning complete" screen after the admin
     * either taps "Done" (hand device to microscopist later) or "Continue
     * Phase 2 now" (single-person deployment path).
     */
    fun proceedToMicroscopistClaim() {
        _phase.value = Phase.MicroscopistClaim
        _microscopistStep.value = MicroscopistStep.Welcome
    }

    // -- Phase 2 advancement ----------------------------------------------

    suspend fun startMicroscopistClaim() {
        val admin = clinicians.current() ?: throw OnboardingError.AdminNotProvisioned
        audit.write(
            action = AuditAction.MicroscopistClaimStarted,
            actorId = admin.actorId,
            actorRoleAtTime = admin.role,
        )
        _microscopistStep.value = MicroscopistStep.Initials
    }

    suspend fun setMicroscopistInitials(initials: String?) {
        val profile = clinicians.current() ?: throw OnboardingError.AdminNotProvisioned
        clinicians.updateInitials(profile.id, initials)
        _microscopistStep.value = MicroscopistStep.Biometric
    }

    /**
     * Microscopist biometric enrolment. In single-person deployments the
     * admin and microscopist are the same human; we keep two audit entries
     * to preserve the chain-of-custody story.
     *
     * After this returns the wizard advances to the (skippable) orientation
     * pages; [finishOrientation] flips `phase` to [Phase.Complete] once the
     * user dismisses or completes them. Phase stays
     * [Phase.MicroscopistClaim] during orientation so the wizard
     * coordinator continues rendering the in-flight UI instead of swapping
     * to RootScreen.
     */
    suspend fun completeMicroscopistClaim() {
        val profile = clinicians.current() ?: throw OnboardingError.AdminNotProvisioned
        audit.write(
            action = AuditAction.MicroscopistBiometricEnrolled,
            actorId = profile.actorId,
            actorRoleAtTime = profile.role,
            resourceType = "clinician",
            resourceId = profile.actorId,
        )
        audit.write(
            action = AuditAction.MicroscopistClaimCompleted,
            actorId = profile.actorId,
            actorRoleAtTime = profile.role,
            resourceType = "clinician",
            resourceId = profile.actorId,
        )
        _microscopistStep.value = MicroscopistStep.Orientation
    }

    /**
     * Final transition out of the wizard. Called when the microscopist
     * either finishes or skips the three orientation pages. The audit trail
     * for the claim is already written by [completeMicroscopistClaim]; no
     * additional entry is needed for orientation per spec §10.
     */
    fun finishOrientation() {
        _phase.value = Phase.Complete
    }

    /**
     * Reset the in-memory state machine after a `Reset device` operation
     * (spec §10 re-onboarding). `MainActivity` reads `phase` to decide
     * whether to render `OnboardingFlow` vs `RootScreen`, so flipping it
     * back to [Phase.AdminProvisioning] synchronously is what wires the
     * auto-relaunch into the admin wizard.
     *
     * Clinic-level pending fields are cleared too — they belong to the
     * previous provisioning context and would mislead the wizard if a
     * future step read them mid-flow. The canonical clinic record stays
     * on the audit log (`clinic_configured`) per spec §10's chain-of-
     * custody guarantee.
     */
    fun reset() {
        _phase.value = Phase.AdminProvisioning
        _adminStep.value = AdminStep.Language
        _microscopistStep.value = MicroscopistStep.Welcome
        _pendingClinicName.value = null
        _pendingClinicJurisdiction.value = null
        _pendingLawfulBasis.value = null
    }

    // -- Testing bypass ---------------------------------------------------

    /**
     * Forces [phase] to [Phase.Complete] without touching the database.
     * Called from [com.malaria.android.MainActivity] when the
     * `SKIP_ONBOARDING` intent extra is present so UI tests and screenshot
     * automation land directly in `RootScreen` on a clean (empty) store.
     * Only reachable in debug builds.
     */
    fun skipForTesting() {
        _phase.value = Phase.Complete
    }

    // -- Private ----------------------------------------------------------

    private suspend fun recordConsent(type: ConsentType, version: String, value: String) {
        consents.record(
            actorId = pendingActorId(),
            consentType = type,
            documentVersion = version,
            value = value,
            appVersion = BuildEnvironment.appVersion,
        )
    }

    /**
     * During Phase 1 the admin actor doesn't exist as a row yet. We use a
     * placeholder identifier on in-flight consent records; after
     * [completeAdminProvisioning], follow-up records have a real `actorId`.
     * Pre-provisioning consent rows can be re-stamped at end of Phase 1 in
     * a future migration if a deployer wants strict FK integrity; v1
     * accepts the placeholder as a known limitation matching iOS.
     */
    private suspend fun pendingActorId(): String =
        runCatching { clinicians.current() }.getOrNull()?.actorId ?: "pre-provisioning"
}

sealed class OnboardingError(message: String) : Exception(message) {
    data object AdminNotProvisioned : OnboardingError("Admin provisioning has not completed.") {
        private fun readResolve(): Any = AdminNotProvisioned
    }
}
