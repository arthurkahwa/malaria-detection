package com.malaria.android.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.ui.locals.LocalSettingsStore

/**
 * Clinic identity (read-only). Sourced from the `clinic_configured`
 * audit entry via [com.malaria.android.services.SettingsStore]. Read-only
 * per spec §11 — the clinic config can only change via Reset device.
 */
@Composable
fun ClinicSection() {
    val settings = LocalSettingsStore.current
    val name by settings.clinicName.collectAsStateWithLifecycle()
    val jurisdiction by settings.jurisdiction.collectAsStateWithLifecycle()
    val lawfulBasis by settings.lawfulBasis.collectAsStateWithLifecycle()

    SectionScaffold(header = "Clinic", footer = "Clinic identity is set during admin provisioning and can only change via Reset device.") {
        ReadOnlyRow(label = "Name", value = name ?: "—")
        ReadOnlyRow(label = "Jurisdiction", value = jurisdiction ?: "—")
        ReadOnlyRow(label = "Lawful basis", value = lawfulBasis ?: "—")
    }
}

@Composable
internal fun SectionScaffold(
    header: String,
    footer: String? = null,
    content: @Composable () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            text = header.uppercase(),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(start = 4.dp),
        )
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                content()
            }
        }
        if (footer != null) {
            Text(
                text = footer,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 4.dp, end = 4.dp),
            )
        }
    }
}

@Composable
internal fun ReadOnlyRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.End,
            modifier = Modifier.padding(start = 12.dp),
        )
    }
}
