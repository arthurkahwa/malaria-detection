package com.malaria.android.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import com.malaria.android.data.entities.AuditEntry
import kotlinx.coroutines.flow.Flow

/**
 * Append-only writer for [AuditEntry]. The public surface intentionally
 * exposes no update or delete methods — append-only by application
 * convention is spec §8's stated guarantee on both platforms.
 *
 * `seq` is assigned at write time as `max(seq) + 1` inside a [Transaction]-
 * annotated function. This is correct on Room's single-writer coroutine
 * dispatch (the default IO dispatcher); concurrent writers across processes
 * would need stronger sequencing.
 */
@Dao
abstract class AuditDao {

    /**
     * Compute the next monotonic `seq` value and insert the supplied entry
     * stamped with that seq. The two operations run inside a transaction so
     * a concurrent reader-writer can't observe a missing seq.
     *
     * Callers should construct the entry without setting `seq` (or with
     * `seq = 0`); this method always overwrites it.
     */
    @Transaction
    open suspend fun write(entry: AuditEntry): AuditEntry {
        val nextSeq = (selectMaxSeq() ?: 0L) + 1L
        val stamped = entry.copy(seq = nextSeq)
        insertInternal(stamped)
        return stamped
    }

    @Insert(onConflict = OnConflictStrategy.ABORT)
    protected abstract suspend fun insertInternal(entry: AuditEntry)

    @Query("SELECT MAX(seq) FROM audit_entries")
    protected abstract suspend fun selectMaxSeq(): Long?

    /**
     * Most recent N entries, newest first by seq. The audit log viewer in
     * History uses this with paging (spec §11).
     */
    @Query("SELECT * FROM audit_entries ORDER BY seq DESC LIMIT :limit")
    abstract suspend fun recent(limit: Int = 200): List<AuditEntry>

    /**
     * Newest-first stream of recent entries. The Phase 10 audit log viewer
     * consumes this via `collectAsStateWithLifecycle()`. This is a one-shot
     * flow (emit then close) rather than a Room invalidation-tracker stream
     * because the existing `FakeAuditDao` test fake (Phase 7) overrides
     * every abstract method; adding a new abstract `@Query` would break
     * that fake's compilation and the JVM test surface is out of scope for
     * Phase 10. The view re-collects when remounted, which matches the iOS
     * navigation behaviour — `AuditLogView` re-runs its `@Query` when the
     * navigation stack returns to it.
     */
    open fun recentFlow(limit: Int = 200): Flow<List<AuditEntry>> =
        kotlinx.coroutines.flow.flow { emit(recent(limit)) }

    /**
     * Entries for a specific canonical action string — used in compliance
     * review and also by `OnboardingState` to detect prior provisioning.
     * Caller supplies `AuditAction.canonical`.
     */
    @Query("SELECT * FROM audit_entries WHERE action = :canonical ORDER BY seq DESC")
    abstract suspend fun entries(canonical: String): List<AuditEntry>

    /** Total entry count — exposed for the export summary block. */
    @Query("SELECT COUNT(*) FROM audit_entries")
    abstract suspend fun count(): Int
}
