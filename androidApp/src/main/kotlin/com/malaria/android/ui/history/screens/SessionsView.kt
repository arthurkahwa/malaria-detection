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
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.ui.history.HistoryDestination
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.history.components.SessionRowView
import com.malaria.android.ui.history.components.SessionStats
import com.malaria.android.ui.locals.LocalPredictionStore

/**
 * Sessions list. Mirrors `iosApp/Views/History/SessionsView.swift`.
 *
 * Pulls every recent prediction from [LocalPredictionStore] and groups
 * client-side via [SessionStats.grouped]; the StateFlow is already hot
 * so this avoids a second Room query. v1.1 will move the aggregator into
 * shared Kotlin (spec §13).
 */
@Composable
fun SessionsView(navigator: HistoryNavigator) {
    val store = LocalPredictionStore.current
    val all by store.recent.collectAsStateWithLifecycle()
    val sessions = SessionStats.grouped(all)

    if (sessions.isEmpty()) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = 16.dp, end = 16.dp),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "No sessions yet",
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    text = "Captured cells will be grouped into sessions here.",
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
            items(sessions, key = { it.sessionId }) { stats ->
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            navigator.push(HistoryDestination.SessionDetail(stats.sessionId))
                        }
                        .padding(top = 4.dp, bottom = 4.dp),
                ) {
                    SessionRowView(stats = stats)
                    HorizontalDivider(modifier = Modifier.padding(top = 4.dp))
                }
            }
        }
    }
}
