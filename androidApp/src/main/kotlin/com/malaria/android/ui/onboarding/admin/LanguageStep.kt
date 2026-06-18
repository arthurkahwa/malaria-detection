package com.malaria.android.ui.onboarding.admin

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.malaria.android.data.LanguagePreference
import com.malaria.android.data.OnboardingLanguage
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import kotlinx.coroutines.launch

/**
 * Step 1 of admin provisioning (spec §10): language picker.
 *
 * Persists the chosen language via DataStore Preferences. "Reset device"
 * preserves it per spec §10 re-onboarding.
 *
 * Mirrors `iosApp/Views/Onboarding/Admin/LanguageStep.swift`.
 */
@Composable
fun LanguageStep() {
    val onboarding = LocalOnboardingState.current
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val preference = remember(context) { LanguagePreference(context.applicationContext) }

    var selected by remember { mutableStateOf(OnboardingLanguage.English) }

    LaunchedEffect(Unit) {
        selected = preference.get()
    }

    WizardStepContainer(
        title = "Choose language",
        subtitle = "Pick the language for the app. You can change this later from Settings.",
        stepIndicator = "Step 1 of 8",
        primaryLabel = "Continue",
        onPrimary = {
            scope.launch {
                preference.set(selected)
                onboarding.advanceFromLanguage()
            }
        },
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            var expanded by remember { mutableStateOf(false) }
            androidx.compose.foundation.layout.Box(modifier = Modifier.fillMaxWidth()) {
                OutlinedTextField(
                    value = selected.displayName,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Language") },
                    trailingIcon = {
                        IconButton(onClick = { expanded = !expanded }) {
                            Icon(Icons.Filled.ArrowDropDown, contentDescription = "Open language list")
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                )
                DropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false },
                ) {
                    OnboardingLanguage.values().forEach { language ->
                        DropdownMenuItem(
                            text = { Text(language.displayName) },
                            onClick = {
                                selected = language
                                expanded = false
                            },
                        )
                    }
                }
            }
        }
    }
}
