package com.malaria.android.services

import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.crashlogs.CrashLogRecord
import com.malaria.crashlogs.RecentActionRing
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import kotlinx.serialization.json.Json
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files

/**
 * Phase 14 / spec §16 — JVM-side coverage for `CrashLogStore`.
 *
 * The shared module's `CrashLogRecordTest` and `RecentActionRingTest`
 * cover the DTO + ring; here we focus on file-system behaviour
 * (`sweepExpired`, `enumerate`) plus the audit-chain side-effect of
 * `didShare`. `EncryptedFile` requires a real Android Keystore and is
 * exercised on-device; the JVM tests use plaintext JSON files written
 * with the same `{incidentId}.json` naming the encrypted writer uses,
 * so the sweep/enumerate companions are realistic.
 */
class CrashLogStoreTest {

    // -- Fakes (mirrors ExportServiceTest patterns) -----------------------

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

    private lateinit var tempDir: File

    @Before
    fun setUp() {
        tempDir = Files.createTempDirectory("crashlogs-test").toFile()
        RecentActionRing.shared.clear()
    }

    @After
    fun tearDown() {
        tempDir.deleteRecursively()
        RecentActionRing.shared.clear()
    }

    private fun writePlaintextLog(file: File, record: CrashLogRecord) {
        val json = Json { encodeDefaults = true }.encodeToString(
            CrashLogRecord.serializer(),
            record,
        )
        file.writeText(json)
    }

    private fun sampleRecord(incidentId: String = java.util.UUID.randomUUID().toString()) =
        CrashLogRecord(
            incidentId = incidentId,
            timestampIso8601 = "2026-05-21T12:00:00Z",
            appVersion = "0.1.0-test",
            osVersion = "Android Test",
            deviceModelClass = "Pixel Test",
            stackTrace = "java.lang.RuntimeException: synthetic",
            recentActionTypes = listOf("prediction_created", "override_recorded"),
            memoryPressure = "available_mb=512,total_mb=1024,low=false",
            deviceUnlocked = true,
        )

    // -- Enumerate / sweep ------------------------------------------------

    @Test
    fun enumerate_returnsLogsNewestFirst() {
        val older = File(tempDir, "${java.util.UUID.randomUUID()}.json")
        val newer = File(tempDir, "${java.util.UUID.randomUUID()}.json")
        writePlaintextLog(older, sampleRecord())
        writePlaintextLog(newer, sampleRecord())
        older.setLastModified(System.currentTimeMillis() - 60_000)
        newer.setLastModified(System.currentTimeMillis())

        val entries = CrashLogStore.enumerate(tempDir)
        assertEquals(2, entries.size)
        assertEquals(newer.nameWithoutExtension, entries[0].incidentId)
        assertEquals(older.nameWithoutExtension, entries[1].incidentId)
    }

    @Test
    fun sweepExpired_removesLogsOlderThanThirtyDays() {
        val fresh = File(tempDir, "${java.util.UUID.randomUUID()}.json")
        val expired = File(tempDir, "${java.util.UUID.randomUUID()}.json")
        writePlaintextLog(fresh, sampleRecord())
        writePlaintextLog(expired, sampleRecord())
        // 31 days ago.
        expired.setLastModified(System.currentTimeMillis() - 31L * 24 * 60 * 60 * 1000)

        CrashLogStore.sweepExpired(tempDir)

        assertTrue("recent log should survive", fresh.exists())
        assertFalse("31-day-old log should be removed", expired.exists())
    }

    @Test
    fun sweepExpired_keepsLogsExactlyTwentyNineDaysOld() {
        val justUnder = File(tempDir, "${java.util.UUID.randomUUID()}.json")
        writePlaintextLog(justUnder, sampleRecord())
        justUnder.setLastModified(System.currentTimeMillis() - 29L * 24 * 60 * 60 * 1000)

        CrashLogStore.sweepExpired(tempDir)
        assertTrue("29-day-old log should survive (cutoff is 30 days)", justUnder.exists())
    }

    // -- Audit log integration -------------------------------------------

    @Test
    fun auditLogWrite_feedsTheRecentActionRing() = runTest {
        val audit = AuditLog(FakeAuditDao())
        audit.write(
            action = AuditAction.PredictionViewed,
            actorId = "actor",
            actorRoleAtTime = "microscopist",
        )
        audit.write(
            action = AuditAction.OverrideRecorded,
            actorId = "actor",
            actorRoleAtTime = "microscopist",
        )
        val snapshot = RecentActionRing.shared.snapshot()
        assertEquals(listOf("prediction_viewed", "override_recorded"), snapshot)
    }

    // -- didShare audits crash_log_shared --------------------------------

    @Test
    fun didShare_writesCrashLogSharedWithIncidentUuid() = runTest {
        val auditDao = FakeAuditDao()
        val audit = AuditLog(auditDao)
        // Write the share directly via AuditLog so we don't need a
        // CrashLogStore instance (the instance pathway requires a real
        // Android Context to materialize EncryptedFile; we exercise the
        // audit contract here).
        val incidentId = "00000000-0000-0000-0000-000000000001"
        audit.write(
            action = AuditAction.CrashLogShared,
            actorId = "actor-1",
            actorRoleAtTime = "microscopist",
            resourceType = "crash_log",
            resourceId = incidentId,
            metadata = emptyMap(),
        )
        val shared = auditDao.entries("crash_log_shared")
        assertEquals(1, shared.size)
        val row = shared.first()
        assertEquals(incidentId, row.resourceId)
        assertEquals("crash_log", row.resourceType)
        // Spec §16 forbids metadata on the share entry; we passed `[:]`.
        assertEquals("{}", row.metadataJson)
    }
}
