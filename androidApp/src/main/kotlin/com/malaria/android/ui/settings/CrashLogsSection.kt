package com.malaria.android.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.ui.locals.LocalCrashLogStore

/**
 * Crash logs section (spec §11 / §16). Phase 14 wires this to the on-device
 * [com.malaria.android.services.CrashLogStore]; the previous Phase 11 stub
 * showed a disabled placeholder.
 */
@Composable
fun CrashLogsSection(navigator: SettingsNavigator) {
    val store = LocalCrashLogStore.current
    val entries by store.entries.collectAsStateWithLifecycle()

    SectionScaffold(
        header = "Crash logs",
        footer = "If the app crashes, a diagnostic log is saved on this device only. Nothing is sent automatically. You can review and share individual logs above. Logs auto-expire after 30 days.",
    ) {
        ReadOnlyRow(label = "Crash log count", value = entries.size.toString())
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            OutlinedButton(
                onClick = { navigator.push(SettingsDestination.CrashLogs) },
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 10.dp),
            ) {
                Icon(
                    imageVector = Icons.Filled.BugReport,
                    contentDescription = null,
                    modifier = Modifier.padding(end = 8.dp),
                )
                Text("Review and share", style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}
