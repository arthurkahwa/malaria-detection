package com.malaria.android.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.ModelDownloadState
import com.malaria.android.ui.locals.LocalModelDownloadService
import com.malaria.android.ui.locals.LocalModelRegistry
import com.malaria.android.ui.locals.LocalSettingsStore
import com.malaria.registry.ModelRegistryEntry

/**
 * Models section (spec §11). Mirrors `ModelsSection.swift`. Shows bundled,
 * downloaded, and available models with real download counts and a
 * working "Clear all caches" button.
 */
@Composable
fun ModelsSection() {
    val modelRegistry = LocalModelRegistry.current
    val modelDownloadService = LocalModelDownloadService.current
    val settings = LocalSettingsStore.current
    val downloadStates by modelDownloadService.downloadStates.collectAsStateWithLifecycle()

    var entries by remember { mutableStateOf<List<ModelRegistryEntry>>(emptyList()) }
    var showClearDialog by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        entries = runCatching { modelRegistry.registry().all() }.getOrDefault(emptyList())
    }

    val bundled = entries.filter { it.bundled }
    val available = entries.filter { !it.bundled }
    val downloaded = available.filter { downloadStates[it.id] is ModelDownloadState.Downloaded }
    val cacheSizeMb = modelDownloadService.totalCacheSizeMb
    val hasCache = downloaded.isNotEmpty()

    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        SectionScaffold(
            header = "Models — Bundled",
            footer = "Bundled models ship inside the app and run without network access.",
        ) {
            bundled.forEach { entry ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(modifier = Modifier.padding(end = 12.dp)) {
                        Text(entry.displayName, style = MaterialTheme.typography.bodyMedium)
                        Text(
                            text = entry.id,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = "Available offline",
                        tint = Color(0xFF2E7D32),
                    )
                }
            }
        }

        SectionScaffold(
            header = "Models — Downloaded",
            footer = "Downloaded models are stored in the app sandbox and survive app restarts.",
        ) {
            if (downloaded.isEmpty()) {
                Text(
                    text = "No downloaded models",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                downloaded.forEach { entry ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Column {
                            Text(entry.displayName, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                text = String.format("%.1f MB", entry.androidFileSizeMb),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Icon(
                            imageVector = Icons.Default.CheckCircle,
                            contentDescription = "Downloaded",
                            tint = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }
        }

        SectionScaffold(header = "Models — Available") {
            available.forEach { entry ->
                Column(modifier = Modifier.fillMaxWidth()) {
                    Text(entry.displayName, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        text = "Requires internet · ${String.format("%.1f MB", entry.androidFileSizeMb)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(
                    text = if (hasCache) "Cache: ${String.format("%.1f MB", cacheSizeMb)}"
                    else "Total cache size: 0 MB",
                    style = MaterialTheme.typography.bodySmall,
                )
                OutlinedButton(
                    onClick = { showClearDialog = true },
                    enabled = hasCache,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Clear all caches")
                }
                if (!hasCache) {
                    Text(
                        text = "No cached models.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }

    if (showClearDialog) {
        AlertDialog(
            onDismissRequest = { showClearDialog = false },
            title = { Text("Clear all caches?") },
            text = { Text("All downloaded models will be deleted. The default model will revert to BN + LeakyReLU (Keras).") },
            confirmButton = {
                TextButton(onClick = {
                    modelDownloadService.clearAllCaches(settings)
                    showClearDialog = false
                }) { Text("Clear") }
            },
            dismissButton = {
                TextButton(onClick = { showClearDialog = false }) { Text("Cancel") }
            },
        )
    }
}
