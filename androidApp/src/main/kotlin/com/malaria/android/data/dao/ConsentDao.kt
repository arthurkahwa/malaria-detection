package com.malaria.android.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.malaria.android.data.entities.ConsentRecord
import com.malaria.android.data.entities.ConsentType
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant

/**
 * Persists onboarding acknowledgements (spec §10). Each record pairs with an
 * [com.malaria.android.data.entities.AuditEntry] recording the same event.
 *
 * Mirrors `iosApp/Persistence/ConsentRepository.swift`.
 */
@Dao
abstract class ConsentDao {

    /**
     * Insert a consent record. Caller passes the canonical [ConsentType]
     * value; the DAO converts to its canonical string.
     */
    suspend fun record(
        actorId: String,
        consentType: ConsentType,
        documentVersion: String,
        value: String,
        appVersion: String,
        timestamp: Instant = Clock.System.now(),
    ): ConsentRecord {
        val record = ConsentRecord(
            timestamp = timestamp,
            actorId = actorId,
            consentType = consentType.canonical,
            documentVersion = documentVersion,
            value = value,
            appVersion = appVersion,
        )
        insert(record)
        return record
    }

    @Insert(onConflict = OnConflictStrategy.ABORT)
    protected abstract suspend fun insert(record: ConsentRecord)

    /** All consent records for a clinician, newest-first. */
    @Query("SELECT * FROM consent_records WHERE actor_id = :actorId ORDER BY timestamp DESC")
    abstract suspend fun records(actorId: String): List<ConsentRecord>

    /**
     * True iff the given clinician has acknowledged this consent type at
     * the specified document version. Used during onboarding flow gating.
     */
    suspend fun hasAccepted(
        type: ConsentType,
        version: String,
        actorId: String,
    ): Boolean = hasAcceptedInternal(type.canonical, version, actorId) > 0

    @Query(
        "SELECT COUNT(*) FROM consent_records " +
            "WHERE actor_id = :actorId AND consent_type = :canonical AND document_version = :version",
    )
    protected abstract suspend fun hasAcceptedInternal(
        canonical: String,
        version: String,
        actorId: String,
    ): Int
}
