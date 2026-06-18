package com.malaria.android.ui.history.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.malaria.android.data.entities.Prediction
import com.malaria.android.ui.history.HistoryDestination
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.history.components.PredictionRowView
import com.malaria.android.ui.history.components.SessionStats
import com.malaria.android.ui.locals.LocalDatabase

/**
 * Header stats + label editing + per-prediction list for one session.
 * Spec §11 + §13. Mirrors `iosApp/Views/History/SessionDetailView.swift`.
 *
 * Predictions are oldest-first within the session — the capture order a
 * clinician would have used.
 */
@Composable
fun SessionDetailScreen(sessionId: String, navigator: HistoryNavigator) {
    val dao = LocalDatabase.current.predictionDao()
    val predictions by produceState<List<Prediction>>(initialValue = emptyList(), key1 = sessionId) {
        value = dao.inSession(sessionId)
    }
    val stats = SessionStats.from(predictions)

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp),
    ) {
        if (stats != null) {
            item {
                OverviewSection(stats)
                HorizontalDivider(modifier = Modifier.padding(top = 12.dp, bottom = 12.dp))
            }
            item {
                LabelSection(
                    stats = stats,
                    onEditLabel = {
                        navigator.push(HistoryDestination.SessionRelabel(sessionId))
                    },
                )
                HorizontalDivider(modifier = Modifier.padding(top = 12.dp, bottom = 12.dp))
            }
        } else {
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 24.dp, bottom = 24.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "Session is empty",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item {
            Text(
                text = "Predictions (${predictions.size})",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 8.dp, bottom = 4.dp),
            )
        }
        items(predictions, key = { it.id }) { prediction ->
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        navigator.push(HistoryDestination.PredictionDetail(prediction.id))
                    }
                    .padding(top = 4.dp, bottom = 4.dp),
            ) {
                PredictionRowView(prediction = prediction)
                HorizontalDivider(modifier = Modifier.padding(top = 4.dp))
            }
        }
    }
}

@Composable
private fun OverviewSection(stats: SessionStats) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            text = "Overview",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 12.dp, bottom = 4.dp),
        )
        OverviewRow("Cells", "${stats.count}")
        OverviewRow("Parasitized", "${stats.parasitizedCount}")
        OverviewRow("Gray zone", "${stats.grayZoneCount}")
        OverviewRow("Mean parasitized prob.", stats.meanParasitizedFormatted)
        OverviewRow("Date range", stats.dateRangeLabel)
    }
}

@Composable
private fun OverviewRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
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

@Composable
private fun LabelSection(stats: SessionStats, onEditLabel: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            text = "Label",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = stats.sessionLabel ?: "—",
                style = MaterialTheme.typography.bodyLarge,
                color = if (stats.sessionLabel == null) MaterialTheme.colorScheme.onSurfaceVariant
                else MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.weight(1f),
            )
            TextButton(onClick = onEditLabel) {
                Text("Edit label")
            }
        }
    }
}
