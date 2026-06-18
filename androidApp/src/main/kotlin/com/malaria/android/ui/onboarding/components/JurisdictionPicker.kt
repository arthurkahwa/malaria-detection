package com.malaria.android.ui.onboarding.components

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.malaria.domain.Jurisdiction

/**
 * Material 3 dropdown over the six [Jurisdiction] enum cases. Display
 * strings stay in English (spec §15: jurisdiction labels are English-only
 * because clinic admins routinely cross-reference them with regulatory
 * documents that are themselves English).
 *
 * The persisted value is the `canonical` lowercase_snake form so it matches
 * what the audit log and shared `RetentionPolicy.minimumYears` expect.
 *
 * Mirrors `iosApp/Views/Onboarding/Components/JurisdictionPicker.swift`.
 */
@Composable
fun JurisdictionPicker(
    selection: Jurisdiction,
    onSelectionChange: (Jurisdiction) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }

    androidx.compose.foundation.layout.Box(modifier = modifier.fillMaxWidth()) {
        OutlinedTextField(
            value = displayName(selection),
            onValueChange = {},
            readOnly = true,
            label = { Text("Jurisdiction") },
            trailingIcon = {
                IconButton(onClick = { expanded = !expanded }) {
                    Icon(Icons.Filled.ArrowDropDown, contentDescription = "Open jurisdiction list")
                }
            },
            modifier = Modifier.fillMaxWidth(),
        )
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            options.forEach { (jurisdiction, label) ->
                DropdownMenuItem(
                    text = { Text(label) },
                    onClick = {
                        onSelectionChange(jurisdiction)
                        expanded = false
                    },
                )
            }
        }
    }
}

private val options: List<Pair<Jurisdiction, String>> = listOf(
    Jurisdiction.US_HIPAA to "United States — HIPAA",
    Jurisdiction.EU_GDPR_DE to "EU GDPR (Germany)",
    Jurisdiction.EU_GDPR_FR to "EU GDPR (France)",
    Jurisdiction.EU_GDPR_GENERIC to "EU GDPR (other member state)",
    Jurisdiction.KE_DPA to "Kenya — Data Protection Act",
    Jurisdiction.OTHER to "Other / not listed",
)

private fun displayName(jurisdiction: Jurisdiction): String =
    options.firstOrNull { it.first == jurisdiction }?.second ?: jurisdiction.canonical
