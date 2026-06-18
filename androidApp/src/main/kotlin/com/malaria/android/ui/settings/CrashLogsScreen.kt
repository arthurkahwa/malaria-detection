package com.malaria.android.ui.settings

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.CrashLogEntry
import com.malaria.android.ui.locals.LocalCrashLogStore
import com.malaria.android.ui.locals.LocalDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.DateFormat
import java.util.Date

/**
 * Settings → Crash logs (spec §16).
 *
 * Compose mirror of `iosApp/Views/Settings/CrashLogsScreen.swift`. Lists
 * the on-device crash log files; tapping a row launches an
 * `Intent.ACTION_SEND` via the existing FileProvider (phase 13 / 14 share
 * the same provider authority, extended in `file_paths.xml`).
 */
@Composable
fun CrashLogsScreen() {
    val store = LocalCrashLogStore.current
    val database = LocalDatabase.current
    val context = LocalContext.current
    val entries by store.entries.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()

    LaunchedEffect(Unit) { store.refresh() }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (entries.isEmpty()) {
            Text(
                text = "No crash logs on this device.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "If the app crashes, a diagnostic log is saved on this device only. Nothing is sent automatically.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            return@Column
        }

        Text(
            text = "${entries.size} log${if (entries.size == 1) "" else "s"}",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary,
        )

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            contentPadding = PaddingValues(bottom = 24.dp),
        ) {
            items(entries, key = { it.incidentId }) { entry ->
                CrashLogRow(
                    entry = entry,
                    onShare = {
                        // 1. Build a FileProvider URI for the encrypted blob.
                        //    The intent recipient sees only the file URI;
                        //    decryption is the user's choice (the JSON is
                        //    encrypted at rest, decryption requires the
                        //    Keystore key on this device — sharing exports
                        //    a blob useful only for triage on this device).
                        val authority = "${context.packageName}.fileprovider"
                        val uri = FileProvider.getUriForFile(
                            context,
                            authority,
                            store.fileFor(entry),
                        )
                        val intent = Intent(Intent.ACTION_SEND).apply {
                            type = "application/json"
                            putExtra(Intent.EXTRA_STREAM, uri)
                            putExtra(Intent.EXTRA_SUBJECT, "Malaria Detector crash log ${entry.incidentId}")
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        val chooser = Intent.createChooser(intent, "Share crash log")
                        chooser.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        context.startActivity(chooser)
                        // 2. Record the share. Spec §16: audited as
                        //    `crash_log_shared` with the incident UUID.
                        scope.launch {
                            val profile = withContext(Dispatchers.IO) {
                                runCatching { database.clinicianDao().current() }.getOrNull()
                            }
                            store.didShare(
                                entry = entry,
                                actorId = profile?.actorId ?: "unknown",
                                actorRole = profile?.role ?: "unknown",
                            )
                        }
                    },
                )
            }
        }

        Text(
            text = "Logs auto-expire after 30 days. Sharing is recorded in the audit log.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun CrashLogRow(
    entry: CrashLogEntry,
    onShare: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        onClick = onShare,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                text = formatAbsolute(entry.timestampMillis),
                style = MaterialTheme.typography.bodyLarge,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = entry.incidentId.take(8),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = "${entry.sizeBytes} B",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

private fun formatAbsolute(millis: Long): String {
    val fmt = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
    return fmt.format(Date(millis))
}
