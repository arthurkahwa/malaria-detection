package com.malaria.android.services

import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.dao.PredictionDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.data.entities.Prediction
import android.content.ContextWrapper
import com.malaria.domain.OverrideContext
import com.malaria.domain.OverrideReason
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.UUID

/**
 * Phase 8 Android JVM tests covering the live-override write path and
 * the camera state-machine error contract. Mirrors the iOS-side
 * `ActiveScreeningTests.swift` so the spec §12 live-override behaviour
 * is asserted identically on both platforms.
 *
 * Uses in-memory fake DAOs in the same shape as Phase 9's
 * [com.malaria.android.ui.override.ReviewOverrideStateTest] — Room
 * requires an Android Context, which would push these tests into
 * `androidTest` and slow the inner loop. The override write path and
 * the override-reason canonical-string contract are persistence- and
 * Compose-agnostic.
 *
 * The CameraX path itself is not exercised here (no
 * `ProcessCameraProvider`, no `LifecycleOwner`); the test asserts the
 * state-machine guard `CameraError.SessionNotRunning` is thrown when
 * `captureOneFrame()` is called before `start()` has bound the use
 * cases.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class LiveOverrideStateTest {

    // -- Fakes ------------------------------------------------------------

    private class FakePredictionDao : PredictionDao {
        private val rows = mutableListOf<Prediction>()

        override suspend fun insert(prediction: Prediction) {
            rows.add(prediction)
        }

        override suspend fun recent(limit: Int): List<Prediction> =
            rows.sortedByDescending { it.timestamp }.take(limit)

        override fun recentFlow(limit: Int): Flow<List<Prediction>> = flow {
            emit(recent(limit))
        }

        override suspend fun inSession(sessionId: String): List<Prediction> =
            rows.filter { it.sessionId == sessionId }.sortedBy { it.timestamp }

        override suspend fun flaggedForReview(): List<Prediction> =
            rows.filter { it.flaggedForReview && it.clinicianOverride == null }
                .sortedByDescending { it.timestamp }

        override fun flaggedForReviewFlow(): Flow<List<Prediction>> = flow {
            emit(flaggedForReview())
        }

        override suspend fun byId(id: String): Prediction? = rows.firstOrNull { it.id == id }

        override suspend fun mostRecent(): Prediction? =
            rows.maxByOrNull { it.timestamp }

        override suspend fun recordOverride(id: String, verdict: String, context: String) {
            val index = rows.indexOfFirst { it.id == id }
            if (index >= 0) {
                rows[index] = rows[index].copy(
                    clinicianOverride = verdict,
                    overrideContext = context,
                )
            }
        }

        override suspend fun markAsDuplicate(duplicateId: String, originalId: String) {
            val index = rows.indexOfFirst { it.id == duplicateId }
            if (index >= 0) {
                rows[index] = rows[index].copy(duplicateOfId = originalId)
            }
        }

        override suspend fun relabel(sessionId: String, label: String?) {
            for (i in rows.indices) {
                if (rows[i].sessionId == sessionId) {
                    rows[i] = rows[i].copy(sessionLabel = label)
                }
            }
        }
    }

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
        val predictions = FakePredictionDao()
        val clinicians = FakeClinicianDao()
        val auditDao = FakeAuditDao()
        val audit = AuditLog(auditDao)
        val scope = TestScope(UnconfinedTestDispatcher())
        val store = PredictionStore(
            dao = predictions,
            audit = audit,
            clinicians = clinicians,
            scope = scope,
        )
    }

    // -- Tests ------------------------------------------------------------

    /**
     * Spec §12: a live override writes the `clinician_override` and
     * `override_context = "live"` columns on the prediction row, plus a
     * single `override_recorded` audit entry that carries
     * `overrideContext = "live"`, the canonical reason, and
     * `contextReviewed = null` (no full-session-context affirmation
     * during active screening). No notes, no second-reviewer initials.
     */
    @Test
    fun liveOverrideRoundtrip_writesContextLiveAndContextReviewedNull() = runTest {
        val fx = Fixture()
        fx.clinicians.enroll(role = "microscopist", initials = "JM")

        val prediction = samplePrediction(parasitizedProb = 0.62, flagged = true)
        fx.predictions.insert(prediction)

        fx.store.override(
            prediction = prediction,
            verdict = "Uninfected",
            context = OverrideContext.LIVE.canonical,
            reason = OverrideReason.IMAGE_QUALITY.canonical,
            notes = null,
            actorInitials = null,
            contextReviewed = null,
        )

        // Prediction columns updated.
        val stored = fx.predictions.byId(prediction.id)
        assertNotNull(stored)
        assertEquals("Uninfected", stored!!.clinicianOverride)
        assertEquals("live", stored.overrideContext)

        // Single override_recorded audit entry with the live-override payload.
        val entries = fx.auditDao.entries(AuditAction.OverrideRecorded.canonical)
        assertEquals(1, entries.size)
        val entry = entries.first()
        assertEquals(prediction.id, entry.resourceId)
        assertEquals("live", entry.overrideContext)
        assertEquals("image_quality", entry.overrideReason)
        assertNull(entry.overrideNotes)
        assertNull(entry.overrideActorInitials)
        assertNull(entry.contextReviewed)
    }

    /**
     * Spec §5 / §8: every override reason persists as its canonical
     * lowercase-snake string and the override context persists as
     * "live" or "review". A translator accidentally re-routing these
     * through a localised resource would break audit-log
     * cross-platform comparability — the assertions here pin the
     * canonical surface.
     */
    @Test
    fun canonicalReasonMapping_isLowercaseSnakeForAllFiveCases() {
        assertEquals("image_quality", OverrideReason.IMAGE_QUALITY.canonical)
        assertEquals("atypical_morphology", OverrideReason.ATYPICAL_MORPHOLOGY.canonical)
        assertEquals("model_false_positive", OverrideReason.MODEL_FALSE_POSITIVE.canonical)
        assertEquals("model_false_negative", OverrideReason.MODEL_FALSE_NEGATIVE.canonical)
        assertEquals("other", OverrideReason.OTHER.canonical)
        assertEquals("live", OverrideContext.LIVE.canonical)
        assertEquals("review", OverrideContext.REVIEW.canonical)
        // All five values are covered.
        assertEquals(5, OverrideReason.entries.size)
    }

    /**
     * Spec §11: `captureOneFrame()` is only meaningful while the
     * camera session is bound to a lifecycle. Without a configured
     * `LifecycleOwner` + `ProcessCameraProvider` the service's state
     * is [CameraService.State.Idle] and the call must throw
     * [CameraService.CameraError.SessionNotRunning].
     *
     * This pins the guard so a future refactor doesn't accidentally
     * fall through to the 2-second poll loop on an empty frame
     * store — which would surface as a less-actionable
     * `CaptureTimeout` instead.
     */
    @Test
    fun captureOneFrame_beforeStart_throwsSessionNotRunning() = runTest {
        // Constructor doesn't touch Context; the unstarted-state guard
        // short-circuits captureOneFrame() before any CameraX call lands.
        // A wrapper around a null base context is sufficient because the
        // test path never invokes a Context method.
        val service = CameraService(
            context = ContextWrapper(null),
            scope = TestScope(UnconfinedTestDispatcher()),
        )

        try {
            service.captureOneFrame()
            fail("Expected CameraError.SessionNotRunning")
        } catch (e: CameraService.CameraError.SessionNotRunning) {
            // Expected — the guard short-circuits before any
            // CameraX call lands. We don't assert on `message`
            // because it's already pinned by the sealed-class
            // declaration.
            assertTrue(true)
        } catch (e: Throwable) {
            fail("Expected SessionNotRunning, got ${e::class.simpleName}: ${e.message}")
        }
    }

    // -- Fixtures ---------------------------------------------------------

    private fun samplePrediction(
        sessionId: String = UUID.randomUUID().toString(),
        parasitizedProb: Double = 0.85,
        label: String = "Parasitized",
        threshold: Double = 0.3,
        flagged: Boolean = false,
    ): Prediction = Prediction(
        id = UUID.randomUUID().toString(),
        sessionId = sessionId,
        timestamp = Clock.System.now(),
        modelId = "BNLeaky_Keras",
        modelVersion = "BNLeaky_Keras",
        parasitizedProb = parasitizedProb,
        uninfectedProb = 1.0 - parasitizedProb,
        label = label,
        threshold = threshold,
        flaggedForReview = flagged,
        inferenceMs = 42,
        imageHash = "a".repeat(64),
        appVersion = "0.1.0-test",
        osVersion = "test",
    )
}
