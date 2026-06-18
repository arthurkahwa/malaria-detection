// Requires API 36 Android emulator. Run via: ./gradlew :androidApp:connectedDebugAndroidTest
package com.malaria.android.data

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.malaria.android.data.TestSupport.samplePrediction
import com.malaria.android.data.dao.PredictionDao
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

/**
 * Mirrors `iosApp/Tests/PredictionRepositoryTests.swift` test-for-test
 * (spec §20).
 */
@RunWith(AndroidJUnit4::class)
class PredictionDaoTest {

    private lateinit var db: MalariaDatabase
    private lateinit var dao: PredictionDao

    @Before
    fun setUp() {
        db = TestSupport.inMemoryDatabase(TestSupport.context)
        dao = db.predictionDao()
    }

    @After
    fun tearDown() { db.close() }

    @Test
    fun insert_appendsToStore() = runTest {
        val p = samplePrediction()
        dao.insert(p)
        val all = dao.recent()
        assertEquals(1, all.size)
        assertEquals(p.id, all.first().id)
    }

    @Test
    fun recent_returnsNewestFirst() = runTest {
        val now = System.currentTimeMillis()
        dao.insert(samplePrediction(timestampMs = now - 60_000))
        dao.insert(samplePrediction(timestampMs = now))
        dao.insert(samplePrediction(timestampMs = now - 30_000))

        val recent = dao.recent()
        assertEquals(3, recent.size)
        assertEquals(now, recent[0].timestamp.toEpochMilliseconds())
    }

    @Test
    fun inSession_returnsOnlyMatchingSession() = runTest {
        val sessionA = UUID.randomUUID().toString()
        val sessionB = UUID.randomUUID().toString()
        dao.insert(samplePrediction(sessionId = sessionA))
        dao.insert(samplePrediction(sessionId = sessionA))
        dao.insert(samplePrediction(sessionId = sessionB))

        val inA = dao.inSession(sessionA)
        assertEquals(2, inA.size)
        assertTrue(inA.all { it.sessionId == sessionA })
    }

    @Test
    fun flaggedForReview_excludesAlreadyOverridden() = runTest {
        dao.insert(samplePrediction(flaggedForReview = true))
        dao.insert(
            samplePrediction(
                flaggedForReview = true,
                clinicianOverride = "Uninfected",
            ),
        )
        dao.insert(samplePrediction(flaggedForReview = false))

        val flagged = dao.flaggedForReview()
        assertEquals(1, flagged.size)
    }

    @Test
    fun recordOverride_setsFields() = runTest {
        val p = samplePrediction(flaggedForReview = true)
        dao.insert(p)

        dao.recordOverride(p.id, "Uninfected", "review")

        val fetched = dao.byId(p.id)
        assertNotNull(fetched)
        assertEquals("Uninfected", fetched!!.clinicianOverride)
        assertEquals("review", fetched.overrideContext)
    }

    @Test
    fun markAsDuplicate_setsDuplicateOfId() = runTest {
        val original = samplePrediction()
        val duplicate = samplePrediction()
        dao.insert(original)
        dao.insert(duplicate)

        dao.markAsDuplicate(duplicate.id, original.id)

        val fetched = dao.byId(duplicate.id)
        assertEquals(original.id, fetched?.duplicateOfId)
    }

    @Test
    fun relabel_appliesLabelToAllInSession() = runTest {
        val sessionId = UUID.randomUUID().toString()
        dao.insert(samplePrediction(sessionId = sessionId))
        dao.insert(samplePrediction(sessionId = sessionId))

        dao.relabel(sessionId, "morning slide")

        val inSession = dao.inSession(sessionId)
        assertEquals(2, inSession.size)
        assertTrue(inSession.all { it.sessionLabel == "morning slide" })
    }

    @Test
    fun mostRecent_returnsLatestTimestamp() = runTest {
        val now = System.currentTimeMillis()
        dao.insert(samplePrediction(timestampMs = now - 60_000))
        dao.insert(samplePrediction(timestampMs = now))

        val mostRecent = dao.mostRecent()
        assertEquals(now, mostRecent?.timestamp?.toEpochMilliseconds())
    }

    @Test
    fun mostRecent_returnsNullWhenEmpty() = runTest {
        assertNull(dao.mostRecent())
    }
}
