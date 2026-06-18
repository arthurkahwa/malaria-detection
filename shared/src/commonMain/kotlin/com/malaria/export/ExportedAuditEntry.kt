package com.malaria.export

import kotlinx.serialization.Serializable

/**
 * Serializable mirror of the platform `AuditEntry` entity (spec §14). All
 * fields from the persistence row are present in spec declaration order.
 *
 * `metadataJson` is the canonical sorted-key JSON string written by the
 * audit log; it is emitted verbatim (not re-parsed) so the export bundle
 * preserves the audit log's own canonical form.
 */
@Serializable
data class ExportedAuditEntry(
    val id: String,
    val seq: Long,
    val timestamp: String,
    val actorId: String,
    val actorRoleAtTime: String,
    val action: String,
    val resourceType: String?,
    val resourceId: String?,
    val metadataJson: String,
    val overrideContext: String?,
    val overrideReason: String?,
    val overrideNotes: String?,
    val contextReviewed: Boolean?,
    val overrideActorInitials: String?,
    val appVersion: String,
    val osVersion: String,
)
