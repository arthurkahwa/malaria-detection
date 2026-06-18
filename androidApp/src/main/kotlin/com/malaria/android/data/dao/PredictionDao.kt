package com.malaria.android.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.malaria.android.data.entities.Prediction
import kotlinx.coroutines.flow.Flow

/**
 * Persists and queries [Prediction] rows. Query surface mirrors
 * `iosApp/Persistence/PredictionRepository.swift`.
 *
 * Read operations expose either suspend or [Flow]. The reactive
 * [recentFlow] is what `HomeScreen` collects via
 * `collectAsStateWithLifecycle()` per spec §19.
 */
@Dao
interface PredictionDao {

    /**
     * Insert a new prediction. Caller is responsible for computing `label`,
     * `flaggedForReview`, and `sessionId` via the shared [com.malaria.util.Threshold]
     * / [com.malaria.session.SessionGrouping] helpers before constructing
     * the entity. ABORT on duplicate primary key — duplicates are a
     * programming error.
     */
    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insert(prediction: Prediction)

    /** Newest-first fetch, optionally limited. */
    @Query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT :limit")
    suspend fun recent(limit: Int = 50): List<Prediction>

    /** Reactive newest-first stream — used by HomeScreen. */
    @Query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT :limit")
    fun recentFlow(limit: Int = 50): Flow<List<Prediction>>

    /**
     * All predictions in a given session, oldest-first (the order a
     * clinician would have captured them).
     */
    @Query("SELECT * FROM predictions WHERE session_id = :sessionId ORDER BY timestamp ASC")
    suspend fun inSession(sessionId: String): List<Prediction>

    /**
     * Predictions in the gray zone awaiting clinician review. Excludes
     * those already overridden (matches iOS predicate exactly).
     */
    @Query(
        "SELECT * FROM predictions " +
            "WHERE flagged_for_review = 1 AND clinician_override IS NULL " +
            "ORDER BY timestamp DESC",
    )
    suspend fun flaggedForReview(): List<Prediction>

    /**
     * Reactive flagged-for-review stream — used by FlaggedForReviewView
     * (Phase 10). Predicate matches the suspend version exactly so the
     * view and any synchronous caller see the same set.
     */
    @Query(
        "SELECT * FROM predictions " +
            "WHERE flagged_for_review = 1 AND clinician_override IS NULL " +
            "ORDER BY timestamp DESC",
    )
    fun flaggedForReviewFlow(): Flow<List<Prediction>>

    @Query("SELECT * FROM predictions WHERE id = :id LIMIT 1")
    suspend fun byId(id: String): Prediction?

    /**
     * Most recent prediction by timestamp — used by SessionGrouping to
     * decide whether the next capture starts a new session.
     */
    @Query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 1")
    suspend fun mostRecent(): Prediction?

    /**
     * Record a clinician override on an existing prediction. Both `live`
     * and `review` contexts go through here; the caller picks the value
     * per spec §12.
     */
    @Query(
        "UPDATE predictions SET clinician_override = :verdict, override_context = :context " +
            "WHERE id = :id",
    )
    suspend fun recordOverride(id: String, verdict: String, context: String)

    /**
     * Mark a prediction as a duplicate of an earlier one. The duplicate
     * stays in storage (no hard delete in v1 per spec §13).
     */
    @Query("UPDATE predictions SET duplicate_of_id = :originalId WHERE id = :duplicateId")
    suspend fun markAsDuplicate(duplicateId: String, originalId: String)

    /**
     * Set the free-text session label on every prediction in the session
     * (denormalised for query simplicity per spec §13).
     */
    @Query("UPDATE predictions SET session_label = :label WHERE session_id = :sessionId")
    suspend fun relabel(sessionId: String, label: String?)
}
