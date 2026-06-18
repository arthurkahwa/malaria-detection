package com.malaria.android.services

import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.dao.ConsentDao
import com.malaria.android.data.dao.PredictionDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.data.entities.ConsentRecord
import com.malaria.android.data.entities.Prediction
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files

/**
 * JVM unit tests for [ExportService]. The shared `ExportBundleBuilder` is
 * exercised by `:shared:check`; here we cover the Android-side adapter:
 *
 *  - the file lands under the supplied cache root with a spec-compliant
 *    filename;
 *  - the audit chain (`export_initiated` + `export_completed`) is written;
 *  - the bundle ZIP is non-empty.
 *
 * In-memory DAO fakes are used (same pattern as `SettingsStoreTest`) rather
 * than spinning up Room with Robolectric; production parity is defended by
 * the schema-drift JVM test and the instrumented Room DAO tests.
 */
class ExportServiceTest {

    // -- Fakes ------------------------------------------------------------

    private class FakeClinicianDao : ClinicianDao() {
        val rows = mutableListOf<ClinicianProfile>()
        override suspend fun current(): ClinicianProfile? = rows.firstOrNull()
        override suspend fun insert(profile: ClinicianProfile) { rows.add(profile) }
        override suspend fun updateInitials(id: String, initials: String?) {}
        override suspend fun markBiometricEnrolled(id: String) {}
        override suspend fun wipe() { rows.clear() }
        fun seed(): ClinicianProfile {
            val p = ClinicianProfile(
                id = "actor-1",
                actorId = "actor-1",
                role = "microscopist",
                initials = "AB",
                enrolledAt = Clock.System.now(),
                biometricEnrolled = true,
            )
            rows.add(p)
            return p
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

    private class FakePredictionDao : PredictionDao {
        val rows = mutableListOf<Prediction>()
        override suspend fun insert(prediction: Prediction) { rows.add(prediction) }
        override suspend fun recent(limit: Int): List<Prediction> =
            rows.sortedByDescending { it.timestamp }.take(limit)
        override fun recentFlow(limit: Int) = flowOf(rows.toList())
        override suspend fun inSession(sessionId: String): List<Prediction> =
            rows.filter { it.sessionId == sessionId }
        override suspend fun flaggedForReview(): List<Prediction> =
            rows.filter { it.flaggedForReview && it.clinicianOverride == null }
        override fun flaggedForReviewFlow() = flowOf(emptyList<Prediction>())
        override suspend fun byId(id: String): Prediction? = rows.firstOrNull { it.id == id }
        override suspend fun mostRecent(): Prediction? = rows.maxByOrNull { it.timestamp }
        override suspend fun recordOverride(id: String, verdict: String, context: String) {}
        override suspend fun markAsDuplicate(duplicateId: String, originalId: String) {}
        override suspend fun relabel(sessionId: String, label: String?) {}
    }

    private class FakeConsentDao : ConsentDao() {
        override suspend fun insert(record: ConsentRecord) {}
        override suspend fun records(actorId: String): List<ConsentRecord> = emptyList()
        override suspend fun hasAcceptedInternal(
            canonical: String,
            version: String,
            actorId: String,
        ): Int = 0
    }

    // -- Fixture ----------------------------------------------------------

    private lateinit var tempCacheRoot: File

    @Before
    fun setUp() {
        tempCacheRoot = Files.createTempDirectory("export-test").toFile()
    }

    @After
    fun tearDown() {
        tempCacheRoot.deleteRecursively()
    }

    private suspend fun makeService(
        clinicConfigured: Boolean = true,
    ): Triple<ExportService, FakeAuditDao, FakeClinicianDao> {
        val clinicians = FakeClinicianDao().also { it.seed() }
        val auditDao = FakeAuditDao()
        val audit = AuditLog(auditDao)
        val settings = SettingsStore(auditDao, audit, clinicians, languagePreference = null)
        if (clinicConfigured) {
            audit.write(
                action = AuditAction.ClinicConfigured,
                actorId = "admin",
                actorRoleAtTime = "admin",
                metadata = mapOf(
                    "clinic_name" to "Test Clinic",
                    "jurisdiction" to "ke_dpa",
                    "lawful_basis" to "vital_interests",
                ),
            )
        }
        settings.hydrate()
        val service = ExportService(
            predictions = FakePredictionDao(),
            audits = auditDao,
            clinicians = clinicians,
            consents = FakeConsentDao(),
            settings = settings,
            auditLog = audit,
            cacheRoot = tempCacheRoot,
            clock = { Clock.System.now() },
            deviceUuidProvider = { "11111111-2222-3333-4444-555555555555" },
        )
        return Triple(service, auditDao, clinicians)
    }

    // -- Tests ------------------------------------------------------------

    @Test
    fun generateBundle_producesNonEmptyFile() = runTest {
        val (service, _, _) = makeService()
        val file = service.generateBundle()
        assertTrue("bundle file should exist", file.exists())
        assertTrue("bundle should be non-empty (got ${file.length()})", file.length() > 0)
    }

    @Test
    fun generateBundle_filenameMatchesSpecPattern() = runTest {
        val (service, _, _) = makeService()
        val file = service.generateBundle()
        // Spec §14: `malaria-detector-export-{deviceUuid-prefix}-{timestamp}.zip`
        assertTrue("got ${file.name}", file.name.startsWith("malaria-detector-export-"))
        assertTrue("got ${file.name}", file.name.endsWith(".zip"))
        // 8-char prefix follows the literal prefix; the device UUID stub
        // we supplied begins with `11111111`.
        assertTrue(
            "expected '11111111' UUID prefix in ${file.name}",
            file.name.contains("-11111111-"),
        )
    }

    @Test
    fun generateBundle_writesAuditChain() = runTest {
        val (service, auditDao, _) = makeService()
        service.generateBundle()

        val initiated = auditDao.entries(AuditAction.ExportInitiated.canonical)
        val completed = auditDao.entries(AuditAction.ExportCompleted.canonical)
        assertEquals(1, initiated.size)
        assertEquals(1, completed.size)

        // export_completed metadata carries size + signature per spec §14.
        val metadata = completed.first().metadataJson
        assertTrue("expected signature in $metadata", metadata.contains("\"signature\""))
        assertTrue("expected size in $metadata", metadata.contains("\"size\""))
    }

    @Test
    fun generateBundle_writesExportFailedWhenClinicNotConfigured() = runTest {
        val (service, auditDao, _) = makeService(clinicConfigured = false)
        var caught: Throwable? = null
        try {
            service.generateBundle()
        } catch (t: Throwable) {
            caught = t
        }
        assertNotNull("expected ExportException", caught)
        assertEquals(
            1,
            auditDao.entries(AuditAction.ExportFailed.canonical).size,
        )
    }

    @Test
    fun lastErrorFlowReflectsFailure() = runTest {
        val (service, _, _) = makeService(clinicConfigured = false)
        try { service.generateBundle() } catch (_: Throwable) { /* expected */ }
        val err = service.lastError.first()
        assertNotNull(err)
    }
}
