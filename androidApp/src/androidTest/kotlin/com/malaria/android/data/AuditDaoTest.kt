// Requires API 36 Android emulator. Run via: ./gradlew :androidApp:connectedDebugAndroidTest
package com.malaria.android.data

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.services.AuditLog
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Mirrors `iosApp/Tests/AuditRepositoryTests.swift` (spec §20).
 *
 * Tests exercise both the raw [AuditDao] (for monotonic-seq verification)
 * and the [AuditLog] wrapper (for JSON-encoded metadata and the
 * canonical-action-string contract).
 */
@RunWith(AndroidJUnit4::class)
class AuditDaoTest {

    private lateinit var db: MalariaDatabase
    private lateinit var dao: AuditDao
    private lateinit var log: AuditLog

    @Before
    fun setUp() {
        db = TestSupport.inMemoryDatabase(TestSupport.context)
        dao = db.auditDao()
        log = AuditLog(dao)
    }

    @After
    fun tearDown() { db.close() }

    private fun blankEntry(action: AuditAction = AuditAction.SessionUnlocked) =
        AuditEntry(
            seq = 0,
            timestamp = Clock.System.now(),
            actorId = "u",
            actorRoleAtTime = "admin",
            action = action.canonical,
            appVersion = "0.1.0-test",
            osVersion = "Android Test",
        )

    @Test
    fun write_assignsMonotonicSeq() = runTest {
        val a = dao.write(blankEntry())
        val b = dao.write(blankEntry())
        val c = dao.write(blankEntry())

        assertEquals(1L, a.seq)
        assertEquals(2L, b.seq)
        assertEquals(3L, c.seq)
    }

    @Test
    fun write_storesCanonicalActionString() = runTest {
        val entry = log.write(
            action = AuditAction.PredictionCreated,
            actorId = "u",
            actorRoleAtTime = "microscopist",
        )
        assertNotNull(entry)
        // Canonical English regardless of UI locale (spec §5/§8).
        assertEquals("prediction_created", entry!!.action)
    }

    @Test
    fun write_encodesMetadataAsSortedJson() = runTest {
        val entry = log.write(
            action = AuditAction.OverrideRecorded,
            actorId = "u",
            actorRoleAtTime = "admin",
            metadata = mapOf("model_id" to "BNLeaky_Keras", "label" to "Parasitized"),
        )
        assertNotNull(entry)
        // Sorted-keys output is deterministic — important so export bundles
        // (spec §14) are byte-stable across exports of the same content.
        assertEquals(
            "{\"label\":\"Parasitized\",\"model_id\":\"BNLeaky_Keras\"}",
            entry!!.metadataJson,
        )
    }

    @Test
    fun write_emptyMetadata_writesEmptyObject() = runTest {
        val entry = log.write(
            action = AuditAction.SessionUnlocked,
            actorId = "u",
            actorRoleAtTime = "admin",
        )
        assertEquals("{}", entry?.metadataJson)
    }

    @Test
    fun write_overrideFields_persistOnEntry() = runTest {
        val entry = log.write(
            action = AuditAction.OverrideRecorded,
            actorId = "u",
            actorRoleAtTime = "microscopist",
            overrideContext = "review",
            overrideReason = "image_quality",
            overrideNotes = "blurry",
            contextReviewed = true,
            overrideActorInitials = "JM",
        )
        assertNotNull(entry)
        assertEquals("review", entry!!.overrideContext)
        assertEquals("image_quality", entry.overrideReason)
        assertEquals("blurry", entry.overrideNotes)
        assertEquals(true, entry.contextReviewed)
        assertEquals("JM", entry.overrideActorInitials)
    }

    @Test
    fun recent_returnsHighestSeqFirst() = runTest {
        repeat(5) { dao.write(blankEntry()) }

        val recent = dao.recent(limit = 3)
        assertEquals(3, recent.size)
        assertEquals(listOf(5L, 4L, 3L), recent.map { it.seq })
    }

    @Test
    fun entriesForAction_filtersByCanonicalString() = runTest {
        dao.write(blankEntry(AuditAction.SessionUnlocked))
        dao.write(blankEntry(AuditAction.PredictionCreated))
        dao.write(blankEntry(AuditAction.SessionUnlocked))

        val unlocks = dao.entries(AuditAction.SessionUnlocked.canonical)
        assertEquals(2, unlocks.size)
    }

    @Test
    fun count_matchesInsertedRows() = runTest {
        repeat(7) { dao.write(blankEntry()) }
        assertEquals(7, dao.count())
    }
}
