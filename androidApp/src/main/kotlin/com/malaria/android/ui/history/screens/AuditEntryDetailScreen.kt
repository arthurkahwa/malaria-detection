package com.malaria.android.ui.history.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.ui.locals.LocalDatabase
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime
import org.json.JSONObject

/**
 * Full breakdown of one AuditEntry. Mirrors
 * `iosApp/Views/History/AuditEntryDetailView.swift` (spec §11).
 *
 * The `metadataJson` payload is parsed via [JSONObject] (`org.json`,
 * Android's native JSON API — no extra dep) and pretty-printed as
 * key-value rows; for `override_recorded` entries the override-specific
 * columns surface in a dedicated section.
 *
 * The lookup is by-id over the cached recent-200 window — keeps the
 * navigation payload a plain string and avoids parceling Room entities.
 */
@Composable
fun AuditEntryDetailScreen(auditEntryId: String) {
    val dao = LocalDatabase.current.auditDao()
    val entry by produceState<AuditEntry?>(initialValue = null, key1 = auditEntryId) {
        value = dao.recent(limit = 200).firstOrNull { it.id == auditEntryId }
    }

    val loaded = entry
    if (loaded == null) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(text = "Loading…", style = MaterialTheme.typography.bodyMedium)
        }
        return
    }
    Content(entry = loaded)
}

@Composable
private fun Content(entry: AuditEntry) {
    val metadataRows = remember(entry.metadataJson) { parseMetadataRows(entry.metadataJson) }
    val isOverride = entry.action == AuditAction.OverrideRecorded.canonical

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Section("Action") {
            DetailRow("Action", entry.action)
            DetailRow("Sequence", "#${entry.seq}")
            DetailRow("Timestamp (UTC)", entry.timestamp.toString())
            DetailRow("Timestamp (local)", localTimestamp(entry))
        }
        Section("Actor") {
            DetailRow("Actor ID", entry.actorId, mono = true)
            DetailRow("Role at time", entry.actorRoleAtTime)
        }
        Section("Resource") {
            DetailRow("Type", entry.resourceType ?: "—")
            DetailRow("ID", entry.resourceId ?: "—", mono = entry.resourceId != null)
        }
        Section("Environment") {
            DetailRow("App version", entry.appVersion)
            DetailRow("OS version", entry.osVersion)
        }
        if (isOverride) {
            Section("Override") {
                DetailRow("Context", entry.overrideContext ?: "—")
                DetailRow("Reason", entry.overrideReason ?: "—")
                DetailRow("Notes", entry.overrideNotes ?: "—")
                DetailRow("Context reviewed", entry.contextReviewed?.let { if (it) "Yes" else "No" } ?: "—")
                DetailRow("Override initials", entry.overrideActorInitials ?: "—")
            }
        }
        Section("Metadata") {
            if (metadataRows.isEmpty()) {
                Text(
                    text = "(empty)",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                metadataRows.forEach { (key, value) ->
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(top = 4.dp, bottom = 4.dp),
                        verticalAlignment = Alignment.Top,
                    ) {
                        Text(
                            text = key,
                            style = MaterialTheme.typography.labelSmall
                                .copy(fontFamily = FontFamily.Monospace),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            text = value,
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        content()
        HorizontalDivider()
    }
}

@Composable
private fun DetailRow(label: String, value: String, mono: Boolean = false) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(top = 4.dp, bottom = 4.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(text = label, modifier = Modifier.weight(1f))
        Text(
            text = value,
            style = if (mono) {
                MaterialTheme.typography.bodyMedium
                    .copy(fontFamily = FontFamily.Monospace)
            } else {
                MaterialTheme.typography.bodyMedium
            },
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private fun parseMetadataRows(json: String): List<Pair<String, String>> {
    val obj = try {
        JSONObject(json)
    } catch (_: Throwable) {
        return emptyList()
    }
    val keys = obj.keys().asSequence().toList().sorted()
    return keys.map { key ->
        val raw = obj.opt(key)
        val value = when (raw) {
            null -> ""
            JSONObject.NULL -> ""
            else -> raw.toString()
        }
        key to value
    }
}

private fun localTimestamp(entry: AuditEntry): String {
    val local = entry.timestamp.toLocalDateTime(TimeZone.currentSystemDefault())
    return "${local.year}-${local.monthNumber.pad()}-${local.dayOfMonth.pad()} " +
        "${local.hour.pad()}:${local.minute.pad()}:${local.second.pad()}"
}

private fun Int.pad(): String = toString().padStart(2, '0')
