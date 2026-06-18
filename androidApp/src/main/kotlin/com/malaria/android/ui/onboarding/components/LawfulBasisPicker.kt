package com.malaria.android.ui.onboarding.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.malaria.domain.LawfulBasis

/**
 * Material 3 dropdown over the three [LawfulBasis] enum cases, with the
 * one-line explanations mandated by spec §10 step 5. Like the jurisdiction
 * picker, display labels stay in English (spec §15).
 *
 * Mirrors `iosApp/Views/Onboarding/Components/LawfulBasisPicker.swift`.
 */
@Composable
fun LawfulBasisPicker(
    selection: LawfulBasis,
    onSelectionChange: (LawfulBasis) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }

    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        androidx.compose.foundation.layout.Box(modifier = Modifier.fillMaxWidth()) {
            OutlinedTextField(
                value = displayName(selection),
                onValueChange = {},
                readOnly = true,
                label = { Text("Lawful basis") },
                trailingIcon = {
                    IconButton(onClick = { expanded = !expanded }) {
                        Icon(Icons.Filled.ArrowDropDown, contentDescription = "Open lawful-basis list")
                    }
                },
                modifier = Modifier.fillMaxWidth(),
            )
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false },
            ) {
                options.forEach { (basis, label) ->
                    DropdownMenuItem(
                        text = { Text(label) },
                        onClick = {
                            onSelectionChange(basis)
                            expanded = false
                        },
                    )
                }
            }
        }

        Text(
            text = explanation(selection),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private val options: List<Pair<LawfulBasis, String>> = listOf(
    LawfulBasis.EXPLICIT_CONSENT to "Explicit consent",
    LawfulBasis.VITAL_INTERESTS to "Vital interests",
    LawfulBasis.HEALTH_PROVISION to "Provision of health care",
)

private fun displayName(basis: LawfulBasis): String =
    options.firstOrNull { it.first == basis }?.second ?: basis.canonical

/**
 * Per spec §10 step 5: "Lawful basis picker … with one-line explanations."
 */
private fun explanation(basis: LawfulBasis): String = when (basis) {
    LawfulBasis.EXPLICIT_CONSENT -> "Patient has given specific, informed consent for screening."
    LawfulBasis.VITAL_INTERESTS -> "Screening is necessary to protect the life of the patient."
    LawfulBasis.HEALTH_PROVISION -> "Screening is part of medical care this clinic provides."
}
