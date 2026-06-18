package com.malaria.android.services

import com.malaria.android.data.BuildEnvironment
import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.crashlogs.RecentActionRing
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant

/**
 * Thin wrapper around [AuditDao] mirroring `iosApp/Services/AuditLog.swift`
 * (spec §8). The split lets a downstream deployer add chain-hashing
 * (spec §18) by replacing the dao call site without touching every caller.
 *
 * Append-only by application convention — no update or delete API.
 *
 * Side-effect (spec §16): every successful write also pushes the action's
 * canonical string into the shared [RecentActionRing] so a future crash
 * log captures the last 50 audit action types without having to touch
 * the database from the signal handler. The ring is in the shared module
 * to keep the contract identical across iOS and Android.
 */
class AuditLog(private val dao: AuditDao) {

    private val _lastWriteError = MutableStateFlow<Throwable?>(null)

    /**
     * Optional `lastWriteError` lets composables surface a write failure
     * (rare, but the spec is explicit that persistence errors should not
     * be silently swallowed).
     */
    val lastWriteError: StateFlow<Throwable?> = _lastWriteError.asStateFlow()

    suspend fun write(
        action: AuditAction,
        actorId: String,
        actorRoleAtTime: String,
        resourceType: String? = null,
        resourceId: String? = null,
        metadata: Map<String, String> = emptyMap(),
        overrideContext: String? = null,
        overrideReason: String? = null,
        overrideNotes: String? = null,
        contextReviewed: Boolean? = null,
        overrideActorInitials: String? = null,
        timestamp: Instant = Clock.System.now(),
    ): AuditEntry? = try {
        val entry = AuditEntry(
            seq = 0, // overwritten by AuditDao.write
            timestamp = timestamp,
            actorId = actorId,
            actorRoleAtTime = actorRoleAtTime,
            action = action.canonical,
            resourceType = resourceType,
            resourceId = resourceId,
            metadataJson = encodeMetadata(metadata),
            overrideContext = overrideContext,
            overrideReason = overrideReason,
            overrideNotes = overrideNotes,
            contextReviewed = contextReviewed,
            overrideActorInitials = overrideActorInitials,
            appVersion = BuildEnvironment.appVersion,
            osVersion = BuildEnvironment.osVersion,
        )
        val written = dao.write(entry)
        _lastWriteError.value = null
        // Spec §16: feed the canonical action string into the shared ring
        // so the crash handler can capture the last 50 actions. Ring
        // stores ONLY the action string — no resource id, actor id,
        // metadata, or timestamp — so the privacy invariants of the
        // crash log hold by construction.
        RecentActionRing.shared.push(action.canonical)
        written
    } catch (t: Throwable) {
        _lastWriteError.value = t
        null
    }
}

/**
 * Stable JSON object encoding: keys sorted, primitive string values escaped.
 * Sorted-key output is deterministic so export bundles (spec §14) are
 * byte-stable across exports of the same content — matches the
 * `.sortedKeys` JSONSerialization behavior on iOS exactly.
 */
internal fun encodeMetadata(metadata: Map<String, String>): String {
    if (metadata.isEmpty()) return "{}"
    val sb = StringBuilder("{")
    metadata.entries.sortedBy { it.key }.forEachIndexed { index, (k, v) ->
        if (index > 0) sb.append(',')
        sb.append('"').append(escape(k)).append('"').append(':')
            .append('"').append(escape(v)).append('"')
    }
    sb.append('}')
    return sb.toString()
}

private fun escape(s: String): String {
    val sb = StringBuilder(s.length)
    for (c in s) when (c) {
        '\\' -> sb.append("\\\\")
        '"' -> sb.append("\\\"")
        '\n' -> sb.append("\\n")
        '\r' -> sb.append("\\r")
        '\t' -> sb.append("\\t")
        '\b' -> sb.append("\\b")
        '\u000C' -> sb.append("\\f")
        else -> if (c.code < 0x20) {
            sb.append("\\u").append("%04x".format(c.code))
        } else {
            sb.append(c)
        }
    }
    return sb.toString()
}
