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
import com.malaria.registry.ModelRegistryEntry

/**
 * Picker over the model registry. Reads entries from
 * `ModelRegistryService.registry().all()` and defaults to `BNLeaky_Keras`
 * per spec §7.
 *
 * Mirrors `iosApp/Views/Onboarding/Components/ModelPicker.swift`.
 */
@Composable
fun ModelPicker(
    selectedId: String,
    entries: List<ModelRegistryEntry>,
    onSelectionChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }
    val selected = entries.firstOrNull { it.id == selectedId }

    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        androidx.compose.foundation.layout.Box(modifier = Modifier.fillMaxWidth()) {
            OutlinedTextField(
                value = selected?.displayName ?: selectedId,
                onValueChange = {},
                readOnly = true,
                label = { Text("Default model") },
                trailingIcon = {
                    IconButton(onClick = { expanded = !expanded }) {
                        Icon(Icons.Filled.ArrowDropDown, contentDescription = "Open model list")
                    }
                },
                modifier = Modifier.fillMaxWidth(),
            )
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false },
            ) {
                entries.forEach { entry ->
                    DropdownMenuItem(
                        text = { Text(entry.displayName) },
                        onClick = {
                            onSelectionChange(entry.id)
                            expanded = false
                        },
                    )
                }
            }
        }

        if (selected != null) {
            Text(
                text = selected.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
