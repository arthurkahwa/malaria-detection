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
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.ui.history.HistoryDestination
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.history.components.PredictionRowView
import com.malaria.android.ui.locals.LocalPredictionStore

/**
 * Newest-first list of stored predictions. Excludes duplicates by
 * default per spec §13; a Switch re-introduces them.
 *
 * Mirrors `iosApp/Views/History/RecentPredictionsView.swift`. The Switch
 * filter happens client-side over the StateFlow rather than via a new
 * DAO query — the DAO surface stays in sync with iOS, and the filter is
 * a presentation concern.
 */
@Composable
fun RecentPredictionsView(navigator: HistoryNavigator) {
    val store = LocalPredictionStore.current
    val all by store.recent.collectAsStateWithLifecycle()
    var showDuplicates by remember { mutableStateOf(false) }

    val visible = if (showDuplicates) all else all.filter { it.duplicateOfId == null }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        item {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 12.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Show duplicates",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.weight(1f),
                )
                Switch(
                    checked = showDuplicates,
                    onCheckedChange = { showDuplicates = it },
                )
            }
            HorizontalDivider()
        }

        if (visible.isEmpty()) {
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 24.dp, bottom = 24.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "No predictions yet",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        } else {
            items(visible, key = { it.id }) { prediction ->
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
