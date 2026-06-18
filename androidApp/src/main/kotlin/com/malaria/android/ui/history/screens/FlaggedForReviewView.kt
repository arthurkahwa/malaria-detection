package com.malaria.android.ui.history.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.malaria.android.ui.history.HistoryDestination
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.history.components.PredictionRowView
import com.malaria.android.ui.locals.LocalDatabase

/**
 * Predictions flagged for clinician review and not yet overridden.
 *
 * Mirrors `iosApp/Views/History/FlaggedForReviewView.swift`. Reads via
 * the new `PredictionDao.flaggedForReviewFlow()` so the predicate stays
 * in SQL and matches the existing `PredictionDao.flaggedForReview()` row
 * for row.
 */
@Composable
fun FlaggedForReviewView(navigator: HistoryNavigator) {
    val dao = LocalDatabase.current.predictionDao()
    val flow = remember { dao.flaggedForReviewFlow() }
    val flagged by flow.collectAsState(initial = emptyList())

    if (flagged.isEmpty()) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = 16.dp, end = 16.dp),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "Nothing flagged",
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    text = "Gray-zone predictions awaiting review will appear here.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 8.dp, start = 8.dp, end = 8.dp),
                )
            }
        }
    } else {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = 16.dp, end = 16.dp),
        ) {
            items(flagged, key = { it.id }) { prediction ->
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
}
