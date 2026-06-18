package com.malaria.android.ui.history

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Flag
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.malaria.android.ui.history.screens.AuditEntryDetailScreen
import com.malaria.android.ui.history.screens.AuditLogView
import com.malaria.android.ui.history.screens.DataManagementView
import com.malaria.android.ui.history.screens.FlaggedForReviewView
import com.malaria.android.ui.history.screens.MarkAsDuplicateScreen
import com.malaria.android.ui.history.screens.PredictionDetailScreen
import com.malaria.android.ui.history.screens.RecentPredictionsView
import com.malaria.android.ui.history.screens.SessionDetailScreen
import com.malaria.android.ui.history.screens.SessionRelabelScreen
import com.malaria.android.ui.history.screens.SessionsView
import com.malaria.android.ui.override.ReviewOverrideScreen
import com.malaria.android.ui.settings.ResetDeviceScreen
import com.malaria.android.ui.settings.SettingsNavigator

/**
 * Top-level History screen and back-stack host. Per spec §11, the five
 * subsections render in declared order. Each subsection's destination is
 * pushed onto [HistoryNavigator]; the [Scaffold]'s top app bar shows a
 * back arrow whenever the stack isn't at the root.
 *
 * Mirrors `iosApp/Views/HistoryTab.swift` and its `HistoryRoot` private
 * view. The destination switch sits inline rather than in its own
 * composable because Compose's `when` over a sealed type compiles to a
 * cheaper dispatch than a function-per-destination indirection.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryRoot() {
    val navigator = rememberHistoryNavigator()
    val current = navigator.current

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(currentTitle(current)) },
                navigationIcon = {
                    if (navigator.canGoBack) {
                        IconButton(onClick = { navigator.pop() }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back",
                            )
                        }
                    }
                },
            )
        },
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            when (val destination = current) {
                HistoryDestination.Subsections -> SubsectionsList(
                    onSelect = navigator::push,
                )
                HistoryDestination.RecentPredictions -> RecentPredictionsView(navigator)
                HistoryDestination.FlaggedForReview -> FlaggedForReviewView(navigator)
                HistoryDestination.Sessions -> SessionsView(navigator)
                HistoryDestination.AuditLog -> AuditLogView(navigator)
                HistoryDestination.DataManagement -> DataManagementView(navigator)
                is HistoryDestination.PredictionDetail -> PredictionDetailScreen(
                    predictionId = destination.predictionId,
                    navigator = navigator,
                )
                is HistoryDestination.SessionDetail -> SessionDetailScreen(
                    sessionId = destination.sessionId,
                    navigator = navigator,
                )
                is HistoryDestination.AuditEntryDetail -> AuditEntryDetailScreen(
                    auditEntryId = destination.auditEntryId,
                )
                is HistoryDestination.MarkAsDuplicate -> MarkAsDuplicateScreen(
                    duplicatePredictionId = destination.duplicatePredictionId,
                    navigator = navigator,
                )
                is HistoryDestination.SessionRelabel -> SessionRelabelScreen(
                    sessionId = destination.sessionId,
                    navigator = navigator,
                )
                is HistoryDestination.ReviewOverride -> ReviewOverrideScreen(
                    predictionId = destination.predictionId,
                    navigator = navigator,
                )
                HistoryDestination.ResetDevice -> ResetDeviceScreen(
                    navigator = remember { SettingsNavigator() },
                )
            }
        }
    }
}

private fun currentTitle(destination: HistoryDestination): String = when (destination) {
    HistoryDestination.Subsections -> "History"
    HistoryDestination.RecentPredictions -> "Recent predictions"
    HistoryDestination.FlaggedForReview -> "Flagged for review"
    HistoryDestination.Sessions -> "Sessions"
    HistoryDestination.AuditLog -> "Audit log"
    HistoryDestination.DataManagement -> "Data management"
    is HistoryDestination.PredictionDetail -> "AI Analysis"
    is HistoryDestination.SessionDetail -> "Session"
    is HistoryDestination.AuditEntryDetail -> "Audit entry"
    is HistoryDestination.MarkAsDuplicate -> "Mark as duplicate"
    is HistoryDestination.SessionRelabel -> "Edit label"
    is HistoryDestination.ReviewOverride -> "Review and override"
    HistoryDestination.ResetDevice -> "Reset device"
}

/**
 * Subsection chooser, spec §11 order:
 *   1. Recent predictions
 *   2. Flagged for review
 *   3. Sessions
 *   4. Audit log
 *   5. Data management
 */
@Composable
private fun SubsectionsList(onSelect: (HistoryDestination) -> Unit) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items(subsections) { subsection ->
            SubsectionRow(subsection = subsection, onClick = { onSelect(subsection.destination) })
        }
    }
}

@Composable
private fun SubsectionRow(subsection: SubsectionEntry, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Icon(
                imageVector = subsection.icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
            )
            Text(
                text = subsection.title,
                style = MaterialTheme.typography.titleMedium,
            )
        }
    }
}

private data class SubsectionEntry(
    val title: String,
    val icon: ImageVector,
    val destination: HistoryDestination,
)

private val subsections: List<SubsectionEntry> = listOf(
    SubsectionEntry("Recent predictions", Icons.Default.History, HistoryDestination.RecentPredictions),
    SubsectionEntry("Flagged for review", Icons.Default.Flag, HistoryDestination.FlaggedForReview),
    SubsectionEntry("Sessions", Icons.AutoMirrored.Filled.List, HistoryDestination.Sessions),
    SubsectionEntry("Audit log", Icons.Default.Description, HistoryDestination.AuditLog),
    SubsectionEntry("Data management", Icons.Default.Storage, HistoryDestination.DataManagement),
)
