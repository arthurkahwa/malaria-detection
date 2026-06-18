package com.malaria.android.ui.settings

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
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.ui.locals.LocalDatabase
import com.malaria.android.ui.locals.LocalSettingsStore

/**
 * Inference policy section (spec §11). Threshold / default model / auto-
 * logout are editable for admins (fresh biometric per spec §9) and
 * read-only for microscopists.
 */
@Composable
fun InferenceSection(navigator: SettingsNavigator) {
    val settings = LocalSettingsStore.current
    val database = LocalDatabase.current
    val threshold by settings.threshold.collectAsStateWithLifecycle()
    val model by settings.defaultModelId.collectAsStateWithLifecycle()
    val autoLogout by settings.autoLogoutMinutes.collectAsStateWithLifecycle()

    var role by remember { mutableStateOf("") }
    LaunchedEffect(Unit) {
        role = runCatching { database.clinicianDao().current()?.role }.getOrNull() ?: ""
    }
    val isAdmin = role == "admin"
    val footer = if (isAdmin) {
        "Editing any value requires a fresh biometric prompt."
    } else {
        "Only the device administrator can change inference policy."
    }

    SectionScaffold(header = "Inference", footer = footer) {
        EditableRow(
            label = "Decision threshold",
            value = "%.2f".format(threshold),
            enabled = isAdmin,
            onClick = { navigator.push(SettingsDestination.EditThreshold) },
        )
        EditableRow(
            label = "Default model",
            value = model,
            enabled = isAdmin,
            onClick = { navigator.push(SettingsDestination.EditDefaultModel) },
        )
        EditableRow(
            label = "Auto-logout",
            value = "$autoLogout min",
            enabled = isAdmin,
            onClick = { navigator.push(SettingsDestination.EditAutoLogout) },
        )
    }
}

@Composable
internal fun EditableRow(
    label: String,
    value: String,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .let { if (enabled) it.clickable(onClick = onClick) else it },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(end = 4.dp),
            )
            if (enabled) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
