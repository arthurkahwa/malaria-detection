package com.malaria.android.ui.settings

import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.services.AuditLog
import com.malaria.android.services.OnboardingState
import com.malaria.android.services.ResetDeviceCoordinator
import com.malaria.android.services.SettingsStore
import com.malaria.android.services.encodeMetadata
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests for [SettingsStore] + [ResetDeviceCoordinator]. Mirrors
 * `iosApp/Tests/SettingsTests.swift` test-for-test.
 *
 * Uses lightweight in-memory fakes for the DAOs rather than spinning up
 * Room with Robolectric; production parity is defended by the on-device
 * Room DAO tests and the JVM `SchemaDriftTest`.
 */
class SettingsStoreTest {

    // -- Fakes ------------------------------------------------------------

    private class FakeClinicianDao : ClinicianDao() {
        val rows = mutableListOf<ClinicianProfile>()

        override suspend fun current(): ClinicianProfile? = rows.firstOrNull()

        override suspend fun insert(profile: ClinicianProfile) { rows.add(profile) }

        override suspend fun updateInitials(id: String, initials: String?) {
            val index = rows.indexOfFirst { it.id == id }
            if (index >= 0) rows[index] = rows[index].copy(initials = initials)
        }

        override suspend fun markBiometricEnrolled(id: String) {
            val index = rows.indexOfFirst { it.id == id }
            if (index >= 0) rows[index] = rows[index].copy(biometricEnrolled = true)
        }

        override suspend fun wipe() { rows.clear() }

        /** Public seed helper — `insert` is protected by Room convention. */
        fun seed(
            id: String = "actor-${rows.size + 1}",
            role: String = "admin",
        ): ClinicianProfile {
            val profile = ClinicianProfile(
                id = id,
                actorId = id,
                role = role,
                initials = null,
                enrolledAt = Clock.System.now(),
                biometricEnrolled = true,
            )
            rows.add(profile)
            return profile
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

        override suspend fun insertInternal(entry: AuditEntry) { rows.add(entry) }

        override suspend fun selectMaxSeq(): Long? = rows.maxOfOrNull { it.seq }

        override suspend fun recent(limit: Int): List<AuditEntry> =
            rows.sortedByDescending { it.seq }.take(limit)

        override suspend fun entries(canonical: String): List<AuditEntry> =
            rows.filter { it.action == canonical }.sortedByDescending { it.seq }

        override suspend fun count(): Int = rows.size
    }

    private suspend fun seedAudit(
        dao: FakeAuditDao,
        action: AuditAction,
        metadata: Map<String, String>,
        actorId: String = "x",
    ) {
        dao.write(
            AuditEntry(
                seq = 0,
                timestamp = Clock.System.now(),
                actorId = actorId,
                actorRoleAtTime = "admin",
                action = action.canonical,
                resourceType = null,
                resourceId = null,
                metadataJson = encodeMetadata(metadata),
                overrideContext = null,
                overrideReason = null,
                overrideNotes = null,
                contextReviewed = null,
                overrideActorInitials = null,
                appVersion = "test",
                osVersion = "test",
            ),
        )
    }

    private class Fixture {
        val clinicians = FakeClinicianDao()
        val auditDao = FakeAuditDao()
        val audit = AuditLog(auditDao)
        val settings = SettingsStore(auditDao, audit, clinicians, languagePreference = null)
    }

    // -- Tests ------------------------------------------------------------

    @Test
    fun hydrate_readsClinicNameFromAuditEntry() = runTest {
        val fx = Fixture()
        seedAudit(
            fx.auditDao,
            AuditAction.ClinicConfigured,
            mapOf(
                "clinic_name" to "Kisumu District Health Centre",
                "jurisdiction" to "ke_dpa",
                "lawful_basis" to "vital_interests",
            ),
        )

        fx.settings.hydrate()

        assertEquals("Kisumu District Health Centre", fx.settings.clinicName.value)
        assertEquals("ke_dpa", fx.settings.jurisdiction.value)
        assertEquals("vital_interests", fx.settings.lawfulBasis.value)
    }

    @Test
    fun hydrate_readsInferencePolicyFromAuditEntry() = runTest {
        val fx = Fixture()
        seedAudit(
            fx.auditDao,
            AuditAction.InferencePolicySet,
            mapOf(
                "threshold" to "0.42",
                "default_model" to "EfficientNetB0_Keras",
                "auto_logout_minutes" to "5",
            ),
        )

        fx.settings.hydrate()

        assertEquals(0.42, fx.settings.threshold.value, 1e-9)
        assertEquals("EfficientNetB0_Keras", fx.settings.defaultModelId.value)
        assertEquals(5, fx.settings.autoLogoutMinutes.value)
    }

    @Test
    fun updateThreshold_writesThresholdChangedAuditEntry() = runTest {
        val fx = Fixture()
        fx.clinicians.seed(role = "admin")
        seedAudit(
            fx.auditDao,
            AuditAction.InferencePolicySet,
            mapOf(
                "threshold" to "0.3",
                "default_model" to "BNLeaky_Keras",
                "auto_logout_minutes" to "15",
            ),
        )

        fx.settings.hydrate()
        fx.settings.updateThreshold(0.55)

        val entries = fx.auditDao.entries(AuditAction.ThresholdChanged.canonical)
        assertEquals(1, entries.size)
        val metadata = entries[0].metadataJson
        assertTrue(metadata.contains("\"old_value\""))
        assertTrue(metadata.contains("\"new_value\""))
        assertTrue(metadata.contains("0.55"))
        assertEquals(0.55, fx.settings.threshold.value, 1e-9)
    }

    @Test
    fun updateDefaultModel_writesDefaultModelChangedAuditEntry() = runTest {
        val fx = Fixture()
        fx.clinicians.seed(role = "admin")
        fx.settings.updateDefaultModel("MobileNetV3Large_Keras")

        val entries = fx.auditDao.entries(AuditAction.DefaultModelChanged.canonical)
        assertEquals(1, entries.size)
        assertTrue(entries[0].metadataJson.contains("MobileNetV3Large_Keras"))
        assertEquals("MobileNetV3Large_Keras", fx.settings.defaultModelId.value)
    }

    @Test
    fun resetCoordinator_wipesClinicianButKeepsAudit() = runTest {
        val fx = Fixture()
        val admin1 = fx.clinicians.seed(id = "admin1", role = "admin")
        seedAudit(fx.auditDao, AuditAction.AdminProvisioningCompleted, emptyMap(), actorId = "admin1")
        seedAudit(fx.auditDao, AuditAction.ClinicConfigured, mapOf("clinic_name" to "X"))
        val onboarding = OnboardingState(fx.clinicians, FakeConsentDao(), fx.audit)
        val coordinator = ResetDeviceCoordinator(fx.clinicians, fx.audit, onboarding, fx.settings)

        val auditCountBefore = fx.auditDao.count()
        coordinator.performReset()

        // Clinician row gone; audit row count strictly increased by exactly
        // one new `device_reprovisioned` entry.
        assertNull(fx.clinicians.current())
        assertEquals(auditCountBefore + 1, fx.auditDao.count())
        val reprovisioned = fx.auditDao.entries(AuditAction.DeviceReprovisioned.canonical)
        assertEquals(1, reprovisioned.size)
        assertEquals("admin1", reprovisioned.first().actorId)
    }

    @Test
    fun resetCoordinator_returnsOnboardingToAdminProvisioning() = runTest {
        val fx = Fixture()
        val admin1 = fx.clinicians.seed(id = "admin1", role = "admin")
        val onboarding = OnboardingState(fx.clinicians, FakeConsentDao(), fx.audit)
        onboarding.rehydrate()
        assertEquals(OnboardingState.Phase.Complete, onboarding.phase.value)

        val coordinator = ResetDeviceCoordinator(fx.clinicians, fx.audit, onboarding, fx.settings)
        coordinator.performReset()

        assertEquals(OnboardingState.Phase.AdminProvisioning, onboarding.phase.value)
        assertEquals(OnboardingState.AdminStep.Language, onboarding.adminStep.value)
        assertNull(onboarding.pendingClinicName.value)
        // Audit entry was written before phase flipped.
        assertNotNull(fx.auditDao.entries(AuditAction.DeviceReprovisioned.canonical).firstOrNull())
    }

    /** Minimal ConsentDao fake for OnboardingState wiring. */
    private class FakeConsentDao : com.malaria.android.data.dao.ConsentDao() {
        override suspend fun insert(record: com.malaria.android.data.entities.ConsentRecord) {}
        override suspend fun records(actorId: String) = emptyList<com.malaria.android.data.entities.ConsentRecord>()
        override suspend fun hasAcceptedInternal(canonical: String, version: String, actorId: String) = 0
    }
}
