package com.malaria.android.services

import com.malaria.android.data.BuildEnvironment
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.dao.PredictionDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.Prediction
import com.malaria.session.SessionGrouping
import com.malaria.util.Threshold
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.datetime.Clock
import java.util.UUID
import com.malaria.domain.Prediction as RawPrediction

/**
 * Write-path façade for [Prediction] rows mirroring
 * `iosApp/Services/PredictionStore.swift`. The reactive read surface is the
 * [recent] StateFlow that `HomeScreen` collects via
 * `collectAsStateWithLifecycle()` per spec §19.
 *
 * Spec §6 active-screening flow: the classifier returns a domain
 * [RawPrediction]; [record] computes label/flag/sessionId via the shared
 * helpers and persists the row + matching `prediction_created` audit entry.
 */
class PredictionStore(
    private val dao: PredictionDao,
    private val audit: AuditLog,
    private val clinicians: ClinicianDao,
    scope: CoroutineScope,
) {

    /**
     * Newest-first stream of recent predictions, backed by Room's reactive
     * `Flow` so the store doesn't have to push updates manually after every
     * write. `stateIn` hot-keeps the latest value across recomposition.
     */
    val recent: StateFlow<List<Prediction>> = dao.recentFlow()
        .stateIn(scope, SharingStarted.Eagerly, emptyList())

    /**
     * Persist a new prediction from the shared classifier's output. Computes
     * label, flag, and session id via the shared helpers per spec §6 data
     * flow. Writes the corresponding `prediction_created` audit entry.
     */
    suspend fun record(
        raw: RawPrediction,
        threshold: Double = Threshold.DEFAULT.toDouble(),
    ): Prediction {
        val actor = runCatching { clinicians.current() }.getOrNull()
        val label = Threshold.label(raw.parasitizedProb, threshold.toFloat())
        val flagged = Threshold.shouldFlagForReview(raw.parasitizedProb)

        val priorMostRecent = runCatching { dao.mostRecent() }.getOrNull()
        val sessionId = SessionGrouping.assignSessionId(
            previousPredictionTimestamp = priorMostRecent?.timestamp,
            previousSessionId = priorMostRecent?.sessionId,
            now = Clock.System.now(),
        )

        val entity = Prediction(
            id = UUID.randomUUID().toString(),
            sessionId = sessionId,
            timestamp = Clock.System.now(),
            modelId = raw.modelId,
            modelVersion = ModelHashCache.hash(raw.modelId),
            parasitizedProb = raw.parasitizedProb.toDouble(),
            uninfectedProb = raw.uninfectedProb.toDouble(),
            label = label,
            threshold = threshold,
            flaggedForReview = flagged,
            inferenceMs = raw.inferenceMs.toInt(),
            imageHash = raw.imageHash,
            appVersion = BuildEnvironment.appVersion,
            osVersion = BuildEnvironment.osVersion,
        )
        dao.insert(entity)
        audit.write(
            action = AuditAction.PredictionCreated,
            actorId = actor?.actorId ?: "unknown",
            actorRoleAtTime = actor?.role ?: "unknown",
            resourceType = "prediction",
            resourceId = entity.id,
            metadata = mapOf(
                "model_id" to entity.modelId,
                "label" to entity.label,
                "flagged" to entity.flaggedForReview.toString(),
            ),
        )
        return entity
    }

    /**
     * Record a clinician override. `context` is "live" or "review"
     * (spec §12); the caller picks based on which UI surfaced it.
     */
    suspend fun override(
        prediction: Prediction,
        verdict: String,
        context: String,
        reason: String,
        notes: String? = null,
        actorInitials: String? = null,
        contextReviewed: Boolean? = null,
    ) {
        val actor = runCatching { clinicians.current() }.getOrNull()
        dao.recordOverride(prediction.id, verdict, context)
        audit.write(
            action = AuditAction.OverrideRecorded,
            actorId = actor?.actorId ?: "unknown",
            actorRoleAtTime = actor?.role ?: "unknown",
            resourceType = "prediction",
            resourceId = prediction.id,
            metadata = mapOf("verdict" to verdict),
            overrideContext = context,
            overrideReason = reason,
            overrideNotes = notes,
            contextReviewed = contextReviewed,
            overrideActorInitials = actorInitials,
        )
    }

    suspend fun markAsDuplicate(duplicate: Prediction, originalId: String) {
        val actor = runCatching { clinicians.current() }.getOrNull()
        dao.markAsDuplicate(duplicate.id, originalId)
        audit.write(
            action = AuditAction.PredictionLinkedAsDuplicate,
            actorId = actor?.actorId ?: "unknown",
            actorRoleAtTime = actor?.role ?: "unknown",
            resourceType = "prediction",
            resourceId = duplicate.id,
            metadata = mapOf("original_id" to originalId),
        )
    }

    suspend fun relabel(sessionId: String, label: String?) {
        val actor = runCatching { clinicians.current() }.getOrNull()
        dao.relabel(sessionId, label)
        audit.write(
            action = AuditAction.SessionRelabeled,
            actorId = actor?.actorId ?: "unknown",
            actorRoleAtTime = actor?.role ?: "unknown",
            resourceType = "session",
            resourceId = sessionId,
            metadata = mapOf("label" to (label ?: "")),
        )
    }
}
