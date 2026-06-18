package com.malaria.android.ui.history.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.malaria.android.data.entities.Prediction
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.history.components.PredictionRowView
import com.malaria.android.ui.locals.LocalDatabase
import com.malaria.android.ui.locals.LocalPredictionStore
import kotlinx.coroutines.launch

/**
 * Picker for choosing the original prediction this one duplicates.
 * Spec §13: candidates are the last 50 predictions in the same session,
 * excluding the duplicate itself. Confirmation required.
 *
 * Mirrors `iosApp/Views/History/MarkAsDuplicateView.swift`.
 */
@Composable
fun MarkAsDuplicateScreen(
    duplicatePredictionId: String,
    navigator: HistoryNavigator,
) {
    val dao = LocalDatabase.current.predictionDao()
    val store = LocalPredictionStore.current
    val scope = rememberCoroutineScope()

    val duplicate by produceState<Prediction?>(initialValue = null, key1 = duplicatePredictionId) {
        value = dao.byId(duplicatePredictionId)
    }
    val sessionCandidates by produceState<List<Prediction>>(
        initialValue = emptyList(),
        key1 = duplicate?.sessionId,
    ) {
        val sessionId = duplicate?.sessionId ?: return@produceState
        value = dao.inSession(sessionId)
            .filter { it.id != duplicatePredictionId }
            .sortedByDescending { it.timestamp }
            .take(50)
    }

    var selected by remember { mutableStateOf<Prediction?>(null) }
    val current = duplicate

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp),
    ) {
        Text(
            text = "Choose the original prediction that this one duplicates. " +
                "The duplicate stays in storage — no data is deleted.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 12.dp, bottom = 12.dp),
        )
        HorizontalDivider()

        if (current == null) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(text = "Loading…", style = MaterialTheme.typography.bodyMedium)
            }
            return
        }

        if (sessionCandidates.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 24.dp, bottom = 24.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "No other predictions in this session.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            Text(
                text = "Candidates",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 12.dp, bottom = 4.dp),
            )
            LazyColumn(modifier = Modifier.fillMaxSize()) {
                items(sessionCandidates, key = { it.id }) { candidate ->
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { selected = candidate }
                            .padding(top = 4.dp, bottom = 4.dp),
                    ) {
                        PredictionRowView(prediction = candidate)
                        HorizontalDivider(modifier = Modifier.padding(top = 4.dp))
                    }
                }
            }
        }
    }

    val active = selected
    if (active != null && current != null) {
        AlertDialog(
            onDismissRequest = { selected = null },
            title = { Text("Mark as duplicate?") },
            text = {
                Text(
                    text = "This prediction will be linked as a duplicate of the one " +
                        "captured at ${active.timestamp}.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    selected = null
                    scope.launch {
                        store.markAsDuplicate(duplicate = current, originalId = active.id)
                        navigator.pop()
                    }
                }) { Text("Mark") }
            },
            dismissButton = {
                TextButton(onClick = { selected = null }) { Text("Cancel") }
            },
        )
    }
}
