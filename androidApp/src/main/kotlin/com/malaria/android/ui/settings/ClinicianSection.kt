package com.malaria.android.ui.settings

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.ui.locals.LocalDatabase

/**
 * Clinician profile section (spec §11):
 *  - UUID (read-only, copyable on tap)
 *  - Role (read-only)
 *  - Initials (editable — pushes [SettingsDestination.EditInitials])
 */
@Composable
fun ClinicianSection(navigator: SettingsNavigator) {
    val database = LocalDatabase.current
    val context = LocalContext.current
    var profile by remember { mutableStateOf<ClinicianProfile?>(null) }
    var copied by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        profile = runCatching { database.clinicianDao().current() }.getOrNull()
    }

    val footer = if (copied) "Copied!" else "UUID is the clinician's device-local identifier. It contains no personal data."

    SectionScaffold(header = "Clinician profile", footer = footer) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable {
                    val id = profile?.actorId
                    if (id != null) {
                        copyToClipboard(context, id)
                        copied = true
                    }
                },
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text("UUID", style = MaterialTheme.typography.bodyMedium)
            Text(
                text = profile?.actorId ?: "—",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                modifier = Modifier.padding(start = 12.dp),
            )
        }

        ReadOnlyRow(label = "Role", value = profile?.role?.replaceFirstChar { it.uppercase() } ?: "—")

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { navigator.push(SettingsDestination.EditInitials) },
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text("Initials", style = MaterialTheme.typography.bodyMedium)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = profile?.initials ?: "—",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(end = 4.dp),
                )
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

private fun copyToClipboard(context: Context, value: String) {
    val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager ?: return
    cm.setPrimaryClip(ClipData.newPlainText("Clinician UUID", value))
}
