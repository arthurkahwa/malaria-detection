package com.malaria.android.data.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import kotlinx.datetime.Instant
import java.util.UUID

/**
 * Append-only audit event (spec §8). The DAO surface intentionally exposes
 * no update or delete methods; append-only is an application convention
 * mirrored from `iosApp/Models/AuditEntry.swift`.
 *
 * `seq` is a monotonic per-device sequence assigned by [AuditDao.write] as
 * `max(seq) + 1` inside a transaction. SQLite AUTOINCREMENT was rejected
 * because the iOS SwiftData side computes seq the same way; cross-platform
 * comparability requires identical semantics.
 *
 * Column names are snake_case so the schema-drift JSON export has a stable
 * surface to compare against `docs/SCHEMA.md`.
 */
@Entity(
    tableName = "audit_entries",
    indices = [
        Index("seq", unique = true),
        Index("action"),
        Index("timestamp"),
    ],
)
data class AuditEntry(
    @PrimaryKey
    @ColumnInfo(name = "id")
    val id: String = UUID.randomUUID().toString(),

    @ColumnInfo(name = "seq")
    val seq: Long,

    @ColumnInfo(name = "timestamp")
    val timestamp: Instant,

    @ColumnInfo(name = "actor_id")
    val actorId: String,

    @ColumnInfo(name = "actor_role_at_time")
    val actorRoleAtTime: String,

    @ColumnInfo(name = "action")
    val action: String,

    @ColumnInfo(name = "resource_type")
    val resourceType: String? = null,

    @ColumnInfo(name = "resource_id")
    val resourceId: String? = null,

    @ColumnInfo(name = "metadata_json")
    val metadataJson: String = "{}",

    @ColumnInfo(name = "override_context")
    val overrideContext: String? = null,

    @ColumnInfo(name = "override_reason")
    val overrideReason: String? = null,

    @ColumnInfo(name = "override_notes")
    val overrideNotes: String? = null,

    @ColumnInfo(name = "context_reviewed")
    val contextReviewed: Boolean? = null,

    @ColumnInfo(name = "override_actor_initials")
    val overrideActorInitials: String? = null,

    @ColumnInfo(name = "app_version")
    val appVersion: String,

    @ColumnInfo(name = "os_version")
    val osVersion: String,
)
