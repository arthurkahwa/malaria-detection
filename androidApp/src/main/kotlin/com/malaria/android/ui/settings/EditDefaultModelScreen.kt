package com.malaria.android.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.BiometricPrompter
import com.malaria.android.services.ModelDownloadState
import com.malaria.android.ui.locals.LocalModelDownloadService
import com.malaria.android.ui.locals.LocalModelRegistry
import com.malaria.android.ui.locals.LocalSettingsStore
import com.malaria.registry.ModelRegistryEntry
import kotlinx.coroutines.launch

/**
 * Edit the default model. Bundled models are always selectable. Non-bundled
 * models can be downloaded from Hugging Face; once downloaded they become
 * selectable and can be deleted. Mirrors `EditDefaultModelView.swift`.
 */
@Composable
fun EditDefaultModelScreen(navigator: SettingsNavigator) {
    val settings = LocalSettingsStore.current
    val current by settings.defaultModelId.collectAsStateWithLifecycle()
    val modelRegistry = LocalModelRegistry.current
    val modelDownloadService = LocalModelDownloadService.current
    val downloadStates by modelDownloadService.downloadStates.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var entries by remember { mutableStateOf<List<ModelRegistryEntry>>(emptyList()) }
    var selected by remember { mutableStateOf(current) }
    var saving by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        entries = runCatching { modelRegistry.registry().all() }.getOrDefault(emptyList())
    }
    LaunchedEffect(current) {
        if (!saving) selected = current
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("Default model", style = MaterialTheme.typography.titleMedium)

        LazyColumn(
            modifier = Modifier.weight(1f, fill = false),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            items(entries, key = { it.id }) { entry ->
                if (entry.bundled) {
                    BundledModelRow(
                        entry = entry,
                        isSelected = selected == entry.id,
                        onSelect = { selected = entry.id },
                    )
                } else {
                    val state = downloadStates[entry.id] ?: ModelDownloadState.NotDownloaded
                    NonBundledModelRow(
                        entry = entry,
                        state = state,
                        isSelected = selected == entry.id,
                        onSelect = { selected = entry.id },
                        onDownload = { modelDownloadService.download(entry) },
                        onDelete = {
                            if (selected == entry.id) selected = "BNLeaky_Keras"
                            modelDownloadService.deleteModel(entry.id)
                        },
                    )
                }
            }
        }

        Button(
            onClick = {
                val activity = context as? FragmentActivity ?: run {
                    errorText = "Biometric prompt unavailable on this activity host."
                    return@Button
                }
                saving = true
                errorText = null
                scope.launch {
                    val outcome = BiometricPrompter(activity).prompt(
                        title = "Confirm default-model change",
                        subtitle = "Use your fingerprint, face, or device PIN.",
                    )
                    when (outcome) {
                        BiometricPrompter.Outcome.Success -> {
                            settings.updateDefaultModel(selected)
                            navigator.pop()
                        }
                        is BiometricPrompter.Outcome.Failure -> errorText = outcome.reason
                        BiometricPrompter.Outcome.Cancelled -> {}
                    }
                    saving = false
                }
            },
            enabled = !saving,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (saving) "Saving…" else "Save")
        }
        errorText?.let { Text(it, color = MaterialTheme.colorScheme.error) }
    }
}

@Composable
private fun BundledModelRow(
    entry: ModelRegistryEntry,
    isSelected: Boolean,
    onSelect: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onSelect() }
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(entry.displayName, style = MaterialTheme.typography.bodyMedium)
            Text(
                text = "Bundled",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary,
            )
        }
        if (isSelected) {
            Icon(
                imageVector = Icons.Default.Check,
                contentDescription = "Selected",
                tint = MaterialTheme.colorScheme.primary,
            )
        }
    }
}

@Composable
private fun NonBundledModelRow(
    entry: ModelRegistryEntry,
    state: ModelDownloadState,
    isSelected: Boolean,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
    onDelete: () -> Unit,
) {
    when (state) {
        is ModelDownloadState.NotDownloaded -> {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(entry.displayName, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        text = String.format("%.1f MB", entry.androidFileSizeMb),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                OutlinedButton(onClick = onDownload) { Text("Download") }
            }
        }

        is ModelDownloadState.Downloading -> {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 6.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(entry.displayName, style = MaterialTheme.typography.bodyMedium)
                LinearProgressIndicator(
                    progress = { state.progress.toFloat() },
                    modifier = Modifier.fillMaxWidth(),
                )
                val receivedMb = entry.androidFileSizeMb * state.progress
                Text(
                    text = String.format("%.1f MB of %.1f MB", receivedMb, entry.androidFileSizeMb),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        is ModelDownloadState.Downloaded -> {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onSelect() }
                    .padding(vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(entry.displayName, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        text = "Downloaded",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
                if (isSelected) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = "Selected",
                        tint = MaterialTheme.colorScheme.primary,
                    )
                }
                IconButton(onClick = onDelete) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete",
                        tint = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }

        is ModelDownloadState.Failed -> {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(entry.displayName, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        text = state.message,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
                OutlinedButton(
                    onClick = onDownload,
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Text("Retry")
                }
            }
        }
    }
}
