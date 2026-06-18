package com.malaria.android.ui.history.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.malaria.android.data.entities.Prediction
import kotlinx.datetime.Clock
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Row layout used in RecentPredictions, FlaggedForReview, and the
 * in-session list inside SessionDetail. Surfaces verdict, probability,
 * model id, override status, and a relative + absolute timestamp.
 *
 * Mirrors `iosApp/Views/History/Components/PredictionRowView.swift`.
 */
@Composable
fun PredictionRowView(
    prediction: Prediction,
    modifier: Modifier = Modifier,
) {
    val probabilityPercent = (prediction.parasitizedProb * 100).roundToInt()

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(top = 4.dp, bottom = 4.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = prediction.label,
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                text = "$probabilityPercent%",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f),
            )
            RiskBandIndicator(parasitizedProb = prediction.parasitizedProb)
        }
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = relativeTimestamp(prediction),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "·",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = absoluteTimestamp(prediction),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Row(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Badge(text = prediction.modelId, color = Color(0xFF1565C0))
            if (prediction.clinicianOverride != null) {
                Badge(text = "Overridden", color = Color(0xFF6A1B9A))
            }
            if (prediction.duplicateOfId != null) {
                Badge(text = "Duplicate", color = Color(0xFF424242))
            }
        }
    }
}

@Composable
private fun Badge(text: String, color: Color) {
    Surface(
        shape = RoundedCornerShape(50),
        color = color.copy(alpha = 0.15f),
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall,
            color = color,
            modifier = Modifier.padding(start = 6.dp, end = 6.dp, top = 2.dp, bottom = 2.dp),
        )
    }
}

/**
 * Short relative string: "5 min ago", "2 h ago", "yesterday". Pure-Kotlin
 * formatter that matches the iOS `RelativeDateTimeFormatter(.short)` cases
 * we care about — no Locale-aware breakdown but good enough for clinical
 * row context (the absolute timestamp sits right beside it).
 */
private fun relativeTimestamp(prediction: Prediction): String {
    val now = Clock.System.now()
    val deltaSeconds = (now - prediction.timestamp).inWholeSeconds
    val abs = abs(deltaSeconds)
    val suffix = if (deltaSeconds >= 0) " ago" else " from now"
    return when {
        abs < 60 -> "just now"
        abs < 3600 -> "${abs / 60} min$suffix"
        abs < 86_400 -> "${abs / 3600} h$suffix"
        abs < 7 * 86_400 -> "${abs / 86_400} d$suffix"
        else -> {
            val local = prediction.timestamp.toLocalDateTime(TimeZone.currentSystemDefault())
            "${local.year}-${local.monthNumber.pad()}-${local.dayOfMonth.pad()}"
        }
    }
}

private fun absoluteTimestamp(prediction: Prediction): String {
    val local = prediction.timestamp.toLocalDateTime(TimeZone.currentSystemDefault())
    return "${local.year}-${local.monthNumber.pad()}-${local.dayOfMonth.pad()} " +
        "${local.hour.pad()}:${local.minute.pad()}"
}

private fun Int.pad(): String = toString().padStart(2, '0')
