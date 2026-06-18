package com.malaria.android.services

import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.dao.ConsentDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.data.entities.ConsentRecord
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests for [OnboardingState]. Mirrors the iOS-side
 * `OnboardingStateTests.swift` and `OnboardingFlowTests.swift` so the
 * state-machine contract is asserted identically on both platforms (spec §10).
 *
 * Uses lightweight in-memory fakes for the three DAOs rather than a real
 * Room build — Room requires an Android Context (Robolectric or an
 * emulator) which would slow the unit-test loop. Production parity is
 * defended by the on-device `androidTest` DAO tests and the JVM-side
 * `SchemaDriftTest`; the state-machine logic this test covers is
 * persistence-agnostic.
 */
class OnboardingStateTest {

    // -- Fakes ------------------------------------------------------------

    private class FakeClinicianDao : ClinicianDao() {
        private val rows = mutableListOf<ClinicianProfile>()

        override suspend fun current(): ClinicianProfile? = rows.firstOrNull()

        override suspend fun insert(profile: ClinicianProfile) {
            rows.add(profile)
        }

        override suspend fun updateInitials(id: String, initials: String?) {
            val index = rows.indexOfFirst { it.id == id }
            if (index >= 0) rows[index] = rows[index].copy(initials = initials)
        }

        override suspend fun markBiometricEnrolled(id: String) {
            val index = rows.indexOfFirst { it.id == id }
            if (index >= 0) rows[index] = rows[index].copy(biometricEnrolled = true)
        }

        override suspend fun wipe() { rows.clear() }
    }

    private class FakeConsentDao : ConsentDao() {
        private val rows = mutableListOf<ConsentRecord>()

        override suspend fun insert(record: ConsentRecord) {
            rows.add(record)
        }

        override suspend fun records(actorId: String): List<ConsentRecord> =
            rows.filter { it.actorId == actorId }.sortedByDescending { it.timestamp }

        override suspend fun hasAcceptedInternal(
            canonical: String,
            version: String,
            actorId: String,
        ): Int = rows.count {
            it.actorId == actorId && it.consentType == canonical && it.documentVersion == version
        }
    }

    private class FakeAuditDao : AuditDao() {
        val rows = mutableListOf<AuditEntry>()
        private var nextSeq = 0L

        override suspend fun write(entry: AuditEntry): AuditEntry {
            nextSeq += 1
            val stamped = entry.copy(seq = nextSeq)
            rows.add(stamped)
            return stamped
        }

        override suspend fun insertInternal(entry: AuditEntry) {
            rows.add(entry)
        }

        override suspend fun selectMaxSeq(): Long? = rows.maxOfOrNull { it.seq }

        override suspend fun recent(limit: Int): List<AuditEntry> =
            rows.sortedByDescending { it.seq }.take(limit)

        override suspend fun entries(canonical: String): List<AuditEntry> =
            rows.filter { it.action == canonical }.sortedByDescending { it.seq }

        override suspend fun count(): Int = rows.size
    }

    private class Fixture {
        val clinicians = FakeClinicianDao()
        val consents = FakeConsentDao()
        val auditDao = FakeAuditDao()
        val audit = AuditLog(auditDao)
        val onboarding = OnboardingState(clinicians, consents, audit)
    }

    // -- Tests mirroring `OnboardingStateTests.swift` ---------------------

    @Test
    fun freshDevice_startsAtAdminProvisioning() {
        val fx = Fixture()

        assertEquals(OnboardingState.Phase.AdminProvisioning, fx.onboarding.phase.value)
        assertEquals(OnboardingState.AdminStep.Language, fx.onboarding.adminStep.value)
    }

    @Test
    fun acceptHippocraticLicense_writesConsentAndAdvances() = runTest {
        val fx = Fixture()

        fx.onboarding.acceptHippocraticLicense(version = "v3.0")

        val records = fx.consents.records("pre-provisioning")
        assertEquals(1, records.size)
        assertEquals("hippocratic_license", records.first().consentType)
        assertEquals("v3.0", records.first().documentVersion)
        assertEquals("accepted", records.first().value)
        assertEquals(OnboardingState.AdminStep.DisclaimerAck, fx.onboarding.adminStep.value)
    }

    @Test
    fun completeAdminProvisioning_createsAdminAndAdvancesToInterstitial() = runTest {
        val fx = Fixture()

        val profile = fx.onboarding.completeAdminProvisioning()

        assertEquals("admin", profile.role)
        // Kotlin's ClinicianDao returns an immutable data class snapshot
        // taken before `markBiometricEnrolled`; the live row in the DB has
        // the flag set. Verify via the DAO rather than the returned value
        // (iOS gets this for free via SwiftData's reference-typed model).
        assertTrue(fx.clinicians.current()!!.biometricEnrolled)
        // Per spec §10 step 8 the admin sees a "Device provisioned for
        // [Clinic name]" interstitial before the microscopist phase begins.
        // Phase stays AdminProvisioning until the admin taps through that
        // screen via `proceedToMicroscopistClaim()`.
        assertEquals(OnboardingState.Phase.AdminProvisioning, fx.onboarding.phase.value)
        assertEquals(OnboardingState.AdminStep.ProvisioningComplete, fx.onboarding.adminStep.value)

        // Two audit entries: biometric enrolment + provisioning completed.
        val completed = fx.auditDao.entries(AuditAction.AdminProvisioningCompleted.canonical)
        val biometric = fx.auditDao.entries(AuditAction.AdminBiometricEnrolled.canonical)
        assertEquals(1, completed.size)
        assertEquals(1, biometric.size)
        assertEquals(profile.actorId, completed.first().actorId)
    }

    @Test
    fun completeMicroscopistClaim_advancesToOrientationKeepingPhase() = runTest {
        val fx = Fixture()
        fx.onboarding.completeAdminProvisioning()
        fx.onboarding.proceedToMicroscopistClaim()
        fx.onboarding.startMicroscopistClaim()

        fx.onboarding.completeMicroscopistClaim()

        // Per spec §10 Phase 2: orientation precedes "Begin screening".
        // The wizard stays in MicroscopistClaim until the user dismisses or
        // finishes orientation via `finishOrientation()`.
        assertEquals(OnboardingState.Phase.MicroscopistClaim, fx.onboarding.phase.value)
        assertEquals(OnboardingState.MicroscopistStep.Orientation, fx.onboarding.microscopistStep.value)
        val entries = fx.auditDao.entries(AuditAction.MicroscopistClaimCompleted.canonical)
        assertEquals(1, entries.size)
    }

    @Test
    fun configureClinic_recordsJurisdictionAndLawfulBasisConsents() = runTest {
        val fx = Fixture()

        fx.onboarding.configureClinic(
            name = "Test Clinic",
            jurisdiction = "ke_dpa",
            lawfulBasis = "vital_interests",
            version = "v1",
        )

        val records = fx.consents.records("pre-provisioning")
        assertEquals(2, records.size)
        assertTrue(records.any { it.consentType == "jurisdiction" && it.value == "ke_dpa" })
        assertTrue(records.any { it.consentType == "lawful_basis" && it.value == "vital_interests" })

        val auditEntries = fx.auditDao.entries(AuditAction.ClinicConfigured.canonical)
        assertEquals(1, auditEntries.size)
        assertTrue(auditEntries.first().metadataJson.contains("Test Clinic"))
    }

    @Test
    fun rehydrate_provisionedDevice_jumpsToComplete() = runTest {
        val fx = Fixture()
        fx.onboarding.completeAdminProvisioning()
        fx.onboarding.proceedToMicroscopistClaim()
        fx.onboarding.startMicroscopistClaim()
        fx.onboarding.completeMicroscopistClaim()
        fx.onboarding.finishOrientation()

        // Simulate app relaunch: new OnboardingState reads persisted state.
        val second = OnboardingState(fx.clinicians, fx.consents, fx.audit)
        second.rehydrate()
        assertEquals(OnboardingState.Phase.Complete, second.phase.value)
    }

    // -- Tests mirroring `OnboardingFlowTests.swift` ----------------------

    @Test
    fun finishOrientation_flipsPhaseToComplete() = runTest {
        val fx = Fixture()
        fx.onboarding.completeAdminProvisioning()
        fx.onboarding.proceedToMicroscopistClaim()
        fx.onboarding.startMicroscopistClaim()
        fx.onboarding.completeMicroscopistClaim()
        assertEquals(OnboardingState.Phase.MicroscopistClaim, fx.onboarding.phase.value)

        fx.onboarding.finishOrientation()

        assertEquals(OnboardingState.Phase.Complete, fx.onboarding.phase.value)
    }

    @Test
    fun finishOrientation_writesNoAdditionalAuditEntries() = runTest {
        val fx = Fixture()
        fx.onboarding.completeAdminProvisioning()
        fx.onboarding.proceedToMicroscopistClaim()
        fx.onboarding.startMicroscopistClaim()
        fx.onboarding.completeMicroscopistClaim()
        val before = fx.auditDao.entries(AuditAction.MicroscopistClaimCompleted.canonical).size

        fx.onboarding.finishOrientation()

        val after = fx.auditDao.entries(AuditAction.MicroscopistClaimCompleted.canonical).size
        assertEquals(before, after)
    }

    @Test
    fun configureClinic_setsPendingFieldsForLaterScreens() = runTest {
        val fx = Fixture()
        assertNull(fx.onboarding.pendingClinicName.value)
        assertNull(fx.onboarding.pendingClinicJurisdiction.value)
        assertNull(fx.onboarding.pendingLawfulBasis.value)

        fx.onboarding.configureClinic(
            name = "Kisumu District Health Centre",
            jurisdiction = "ke_dpa",
            lawfulBasis = "health_provision",
            version = "v1.0",
        )

        assertEquals("Kisumu District Health Centre", fx.onboarding.pendingClinicName.value)
        assertEquals("ke_dpa", fx.onboarding.pendingClinicJurisdiction.value)
        assertEquals("health_provision", fx.onboarding.pendingLawfulBasis.value)
    }

    @Test
    fun advanceFromLanguage_movesToWelcome() {
        val fx = Fixture()
        assertEquals(OnboardingState.AdminStep.Language, fx.onboarding.adminStep.value)

        fx.onboarding.advanceFromLanguage()

        assertEquals(OnboardingState.AdminStep.Welcome, fx.onboarding.adminStep.value)
    }

    @Test
    fun advanceFromWelcome_movesToLicenseAck() {
        val fx = Fixture()
        fx.onboarding.advanceFromLanguage()

        fx.onboarding.advanceFromWelcome()

        assertEquals(OnboardingState.AdminStep.LicenseAck, fx.onboarding.adminStep.value)
    }

    @Test
    fun advanceFromLanguage_isNoopWhenNotOnLanguageStep() {
        val fx = Fixture()
        fx.onboarding.advanceFromLanguage() // now on Welcome

        fx.onboarding.advanceFromLanguage()

        // Doesn't skip ahead — guards against double-tap glitches.
        assertEquals(OnboardingState.AdminStep.Welcome, fx.onboarding.adminStep.value)
    }

    @Test
    fun proceedToMicroscopistClaim_switchesPhaseFromAdminToMicroscopist() = runTest {
        val fx = Fixture()
        fx.onboarding.completeAdminProvisioning()
        assertEquals(OnboardingState.Phase.AdminProvisioning, fx.onboarding.phase.value)
        assertEquals(OnboardingState.AdminStep.ProvisioningComplete, fx.onboarding.adminStep.value)

        fx.onboarding.proceedToMicroscopistClaim()

        assertEquals(OnboardingState.Phase.MicroscopistClaim, fx.onboarding.phase.value)
        assertEquals(OnboardingState.MicroscopistStep.Welcome, fx.onboarding.microscopistStep.value)
    }

    /**
     * The composition root mounts [OnboardingState] wizard while phase ≠
     * Complete and `RootScreen` once it is. This test asserts the
     * finish-orientation transition is exactly that gate.
     */
    @Test
    fun rootViewVisibility_followsCompletionGate() = runTest {
        val fx = Fixture()
        assertNotEquals(
            "Fresh device must show onboarding, not RootScreen",
            OnboardingState.Phase.Complete,
            fx.onboarding.phase.value,
        )

        fx.onboarding.completeAdminProvisioning()
        fx.onboarding.proceedToMicroscopistClaim()
        fx.onboarding.startMicroscopistClaim()
        fx.onboarding.completeMicroscopistClaim()
        assertNotEquals(
            "Wizard still owns the screen while orientation pages are visible",
            OnboardingState.Phase.Complete,
            fx.onboarding.phase.value,
        )

        fx.onboarding.finishOrientation()
        assertEquals(
            "RootScreen mounts only after orientation finishes",
            OnboardingState.Phase.Complete,
            fx.onboarding.phase.value,
        )
    }
}
