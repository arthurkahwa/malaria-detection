package com.malaria.android.ui.onboarding.microscopist

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.unit.dp
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import kotlinx.coroutines.launch

/**
 * Phase 2 step 2 (spec §10): optional 2-character initials or skip.
 * Initials appear next to the microscopist's identifier in audit reports so
 * reviewers can recognise who recorded an override.
 *
 * Mirrors `iosApp/Views/Onboarding/Microscopist/InitialsStep.swift`.
 */
@Composable
fun InitialsStep() {
    val onboarding = LocalOnboardingState.current
    val scope = rememberCoroutineScope()

    var initials by remember { mutableStateOf("") }
    val trimmed = initials.trim()

    WizardStepContainer(
        title = "Your initials",
        subtitle = "Up to two characters. These appear next to your overrides in the audit log. Optional — you can skip this.",
        stepIndicator = "Step 2 of 4",
        primaryLabel = "Continue",
        primaryEnabled = trimmed.length <= 2,
        onPrimary = {
            scope.launch {
                onboarding.setMicroscopistInitials(if (trimmed.isEmpty()) null else trimmed)
            }
        },
        secondaryLabel = "Skip",
        onSecondary = {
            scope.launch { onboarding.setMicroscopistInitials(null) }
        },
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Initials", style = MaterialTheme.typography.titleSmall)
            OutlinedTextField(
                value = initials,
                onValueChange = { newValue ->
                    // Clamp to 2 characters as the user types — the spec
                    // column is exactly 2 chars wide.
                    initials = if (newValue.length > 2) newValue.take(2) else newValue
                },
                placeholder = { Text("e.g. JM") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(
                    capitalization = KeyboardCapitalization.Characters,
                    imeAction = ImeAction.Done,
                ),
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}
