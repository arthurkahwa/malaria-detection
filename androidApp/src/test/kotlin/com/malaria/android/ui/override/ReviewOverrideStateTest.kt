package com.malaria.android.ui.override

import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.dao.PredictionDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.data.entities.Prediction
import com.malaria.android.services.AuditLog
import com.malaria.android.services.PredictionStore
import com.malaria.domain.OverrideContext
import com.malaria.domain.OverrideReason
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.UUID

/**
 * Phase 9 review-override JVM tests. Mirrors the iOS-side
 * `ReviewOverrideTests.swift` so the contract is asserted identically on
 * both platforms (spec §12 review override).
 *
 * Uses in-memory fake DAOs in the same shape as
 * [com.malaria.android.services.OnboardingStateTest] — Room requires an
 * Android Context, which would push these tests into `androidTest` and
 * slow the inner-loop. The state-machine logic ([ReviewOverrideValidator])
 * and the audit-write roundtrip ([PredictionStore.override]) are both
 * persistence-agnostic.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ReviewOverrideStateTest {

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

    @Test
    fun saveEnable_requiresVerdictReasonAndCheckbox() {
        // No selection at all.
        assertFalse(ReviewOverrideValidator.canSave(ReviewOverrideUiState()))

        // Verdict only.
        assertFalse(
            ReviewOverrideValidator.canSave(
                ReviewOverrideUiState(verdict = ReviewOverrideValidator.Verdict.Parasitized),
            ),
        )

        // Verdict + reason but not reviewed.
        assertFalse(
            ReviewOverrideValidator.canSave(
                ReviewOverrideUiState(
                    verdict = ReviewOverrideValidator.Verdict.Uninfected,
                    reason = OverrideReason.IMAGE_QUALITY,
                    contextReviewed = false,
                ),
            ),
        )

        // Verdict + reviewed but no reason.
        assertFalse(
            ReviewOverrideValidator.canSave(
                ReviewOverrideUiState(
                    verdict = ReviewOverrideValidator.Verdict.Parasitized,
                    reason = null,
                    contextReviewed = true,
                ),
            ),
        )

        // All three: enabled.
        assertTrue(
            ReviewOverrideValidator.canSave(
                ReviewOverrideUiState(
                    verdict = ReviewOverrideValidator.Verdict.Uninfected,
                    reason = OverrideReason.MODEL_FALSE_POSITIVE,
                    contextReviewed = true,
                ),
            ),
        )
    }

    @Test
    fun overrideRoundtrip_setsFieldsAndWritesAuditEntry() = runTest {
        val fx = Fixture()
        val clinician = fx.clinicians.enroll(role = "microscopist", initials = "JM")
        fx.clinicians.markBiometricEnrolled(clinician.id)

        val prediction = samplePrediction(parasitizedProb = 0.53, flagged = true)
        fx.predictions.insert(prediction)

        fx.store.override(
            prediction = prediction,
            verdict = "Uninfected",
            context = OverrideContext.REVIEW.canonical,
            reason = OverrideReason.MODEL_FALSE_POSITIVE.canonical,
            notes = "Edge artefact on the slide.",
            actorInitials = "AK",
            contextReviewed = true,
        )

        // Prediction columns updated.
        val stored = fx.predictions.byId(prediction.id)
        assertNotNull(stored)
        assertEquals("Uninfected", stored!!.clinicianOverride)
        assertEquals("review", stored.overrideContext)

        // Audit entry written with the full Phase 9 payload.
        val entries = fx.auditDao.entries(AuditAction.OverrideRecorded.canonical)
        assertEquals(1, entries.size)
        val entry = entries.first()
        assertEquals(prediction.id, entry.resourceId)
        assertEquals("review", entry.overrideContext)
        assertEquals("model_false_positive", entry.overrideReason)
        assertEquals("Edge artefact on the slide.", entry.overrideNotes)
        assertEquals("AK", entry.overrideActorInitials)
        assertEquals(true, entry.contextReviewed)
        assertEquals(clinician.actorId, entry.actorId)
    }

    /**
     * Spec §12: "An override cannot be undone in v1." The PredictionDetail
     * affordance hides itself once `clinicianOverride` is set, so a second
     * override is unreachable from the UI — but the underlying write must
     * remain safe if invoked (e.g. a stale screen in another nav stack).
     * Each call appends a new audit entry; the displayed verdict reflects
     * the most recent override only.
     */
    @Test
    fun override_secondCallAppendsAuditAndUpdatesVerdict() = runTest {
        val fx = Fixture()
        fx.clinicians.enroll(role = "microscopist", initials = "JM")

        val prediction = samplePrediction(flagged = true)
        fx.predictions.insert(prediction)

        fx.store.override(
            prediction = prediction,
            verdict = "Uninfected",
            context = "review",
            reason = OverrideReason.IMAGE_QUALITY.canonical,
            notes = null,
            actorInitials = "JM",
            contextReviewed = true,
        )
        fx.store.override(
            prediction = prediction,
            verdict = "Parasitized",
            context = "review",
            reason = OverrideReason.OTHER.canonical,
            notes = "Second reviewer disagrees.",
            actorInitials = "AK",
            contextReviewed = true,
        )

        // Most-recent verdict on the row, two audit entries in the log.
        val stored = fx.predictions.byId(prediction.id)
        assertEquals("Parasitized", stored?.clinicianOverride)
        assertEquals("review", stored?.overrideContext)
        val entries = fx.auditDao.entries(AuditAction.OverrideRecorded.canonical)
        assertEquals(2, entries.size)
    }

    @Test
    fun displayLabels_coverAllFiveCanonicalReasons() {
        // Spec §15 keeps these labels English even when the rest of the
        // app is localised; pinning the strings here means a translator
        // accidentally re-routing them through strings.xml would trip a
        // test rather than silently shipping translated reasons.
        assertEquals(
            "Image quality",
            ReviewOverrideValidator.displayLabel(OverrideReason.IMAGE_QUALITY),
        )
        assertEquals(
            "Atypical morphology",
            ReviewOverrideValidator.displayLabel(OverrideReason.ATYPICAL_MORPHOLOGY),
        )
        assertEquals(
            "Model false-positive",
            ReviewOverrideValidator.displayLabel(OverrideReason.MODEL_FALSE_POSITIVE),
        )
        assertEquals(
            "Model false-negative",
            ReviewOverrideValidator.displayLabel(OverrideReason.MODEL_FALSE_NEGATIVE),
        )
        assertEquals(
            "Other",
            ReviewOverrideValidator.displayLabel(OverrideReason.OTHER),
        )
        // All five canonical values must be covered by `allReasons`.
        assertEquals(5, ReviewOverrideValidator.allReasons.size)
    }

    // -- Fixtures ---------------------------------------------------------

    private fun samplePrediction(
        sessionId: String = UUID.randomUUID().toString(),
        timestamp: Instant = Clock.System.now(),
        parasitizedProb: Double = 0.85,
        label: String = "Parasitized",
        threshold: Double = 0.3,
        flagged: Boolean = false,
    ): Prediction = Prediction(
        id = UUID.randomUUID().toString(),
        sessionId = sessionId,
        timestamp = timestamp,
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
