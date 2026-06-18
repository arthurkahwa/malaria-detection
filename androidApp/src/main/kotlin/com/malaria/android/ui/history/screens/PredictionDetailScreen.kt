package com.malaria.android.ui.history.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.Prediction
import com.malaria.android.ui.history.HistoryDestination
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.history.components.RiskBandIndicator
import com.malaria.android.ui.locals.LocalAuditLog
import com.malaria.android.ui.locals.LocalDatabase
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime
import kotlin.math.roundToInt

/**
 * AI Analysis detail (spec §11 §12). Sections: verdict + risk band,
 * context (timestamps, model, threshold, hash), session link, actions.
 *
 * Writes `prediction_viewed` once on entry. The audit-write is keyed by
 * `predictionId` inside [LaunchedEffect] so a back-and-forward within the
 * same screen does not duplicate the row — only a fresh navigation tap
 * (new key value) re-fires. Mirrors iOS's `didAudit` flag semantics
 * exactly while being purely declarative.
 *
 * The prediction itself is re-fetched via [com.malaria.android.data.dao.PredictionDao.byId]
 * rather than passed through the navigator stack to keep the destination
 * payload a plain `String` id — Parcelable-free navigation (Phase 11 can
 * graduate this to Compose Navigation if needed).
 */
@Composable
fun PredictionDetailScreen(
    predictionId: String,
    navigator: HistoryNavigator,
) {
    val dao = LocalDatabase.current.predictionDao()
    val auditLog = LocalAuditLog.current
    val clinicians = LocalDatabase.current.clinicianDao()

    val prediction by produceState<Prediction?>(initialValue = null, key1 = predictionId) {
        value = dao.byId(predictionId)
    }

    LaunchedEffect(predictionId) {
        val actor = runCatching { clinicians.current() }.getOrNull()
        auditLog.write(
            action = AuditAction.PredictionViewed,
            actorId = actor?.actorId ?: "unknown",
            actorRoleAtTime = actor?.role ?: "unknown",
            resourceType = "prediction",
            resourceId = predictionId,
        )
    }

    val loaded = prediction
    if (loaded == null) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Loading…", style = MaterialTheme.typography.bodyMedium)
        }
        return
    }
    Content(prediction = loaded, navigator = navigator)
}

@Composable
private fun Content(prediction: Prediction, navigator: HistoryNavigator) {
    val clipboard = LocalClipboardManager.current
    var hashCopied by remember { mutableStateOf(false) }
    val probabilityPercent = (prediction.parasitizedProb * 100).roundToInt()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        // --- Verdict
        Section(title = "Verdict") {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = prediction.label,
                    style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.SemiBold),
                    modifier = Modifier.weight(1f),
                )
                Text(
                    text = "$probabilityPercent%",
                    style = MaterialTheme.typography.titleMedium,
                )
            }
            RiskBandIndicator(parasitizedProb = prediction.parasitizedProb)
            if (prediction.clinicianOverride != null) {
                val context = prediction.overrideContext ?: "review"
                Text(
                    text = "Originally ${prediction.label} $probabilityPercent%, overridden to " +
                        "${prediction.clinicianOverride} (in $context)",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFF6A1B9A),
                )
            }
        }

        // --- Context
        Section(title = "Context") {
            DetailRow("Captured (UTC)", prediction.timestamp.toString())
            DetailRow("Captured (local)", localTimestamp(prediction))
            DetailRow("Model", prediction.modelId)
            DetailRow("Model version", prediction.modelVersion)
            DetailRow("Threshold at capture", "%.2f".format(prediction.threshold))
            DetailRow("Inference time", "${prediction.inferenceMs} ms")
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        clipboard.setText(AnnotatedString(prediction.imageHash))
                        hashCopied = true
                    }
                    .padding(top = 8.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(text = "Image hash", modifier = Modifier.weight(1f))
                Text(
                    text = if (hashCopied) "Copied!" else prediction.imageHash.take(16) + "…",
                    style = MaterialTheme.typography.bodySmall
                        .copy(fontFamily = FontFamily.Monospace),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        // --- Session
        Section(title = "Session") {
            prediction.sessionLabel?.takeIf { it.isNotEmpty() }?.let { label ->
                DetailRow("Label", label)
            }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        navigator.push(HistoryDestination.SessionDetail(prediction.sessionId))
                    }
                    .padding(top = 8.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(text = "Open session", modifier = Modifier.weight(1f))
                Text(
                    text = prediction.sessionId.take(8),
                    style = MaterialTheme.typography.bodySmall
                        .copy(fontFamily = FontFamily.Monospace),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        // --- Actions
        Section(title = "Actions") {
            TextButton(
                onClick = {
                    navigator.push(HistoryDestination.MarkAsDuplicate(prediction.id))
                },
                enabled = prediction.duplicateOfId == null,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(
                    imageVector = Icons.Default.Description,
                    contentDescription = null,
                )
                Text(
                    text = "Mark as duplicate of…",
                    modifier = Modifier.padding(start = 8.dp).weight(1f),
                )
            }
            // Spec §12: "An override cannot be undone in v1". Hide the
            // affordance once an override has been recorded so the UI
            // doesn't suggest a second pass is meaningful — re-overrides
            // would write a new audit entry but the displayed verdict
            // only ever reflects the most recent override.
            if (prediction.clinicianOverride == null) {
                TextButton(
                    onClick = {
                        navigator.push(HistoryDestination.ReviewOverride(prediction.id))
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        imageVector = Icons.Default.Edit,
                        contentDescription = null,
                    )
                    Text(
                        text = "Review and override",
                        modifier = Modifier.padding(start = 8.dp).weight(1f),
                    )
                }
            }
        }
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
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
private fun DetailRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(top = 4.dp, bottom = 4.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(text = label, modifier = Modifier.weight(1f))
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private fun localTimestamp(prediction: Prediction): String {
    val local = prediction.timestamp.toLocalDateTime(TimeZone.currentSystemDefault())
    return "${local.year}-${local.monthNumber.pad()}-${local.dayOfMonth.pad()} " +
        "${local.hour.pad()}:${local.minute.pad()}:${local.second.pad()}"
}

private fun Int.pad(): String = toString().padStart(2, '0')
