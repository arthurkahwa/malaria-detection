package com.malaria.android.ui.onboarding.admin

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
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
import com.malaria.android.services.ModelRegistryService
import com.malaria.android.ui.locals.LocalModelRegistry
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.ModelPicker
import com.malaria.android.ui.onboarding.components.ThresholdSlider
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import com.malaria.registry.ModelRegistryEntry
import kotlinx.coroutines.launch

/**
 * Step 6 of admin provisioning (spec §10): inference policy — threshold,
 * default model, auto-logout timeout.
 *
 * Mirrors `iosApp/Views/Onboarding/Admin/InferencePolicyStep.swift`.
 */
@Composable
fun InferencePolicyStep() {
    val onboarding = LocalOnboardingState.current
    val registryService: ModelRegistryService = LocalModelRegistry.current
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var threshold by remember { mutableStateOf(0.3) }
    var selectedModelId by remember { mutableStateOf("BNLeaky_Keras") }
    var autoLogoutMinutes by remember { mutableStateOf(15) }
    var entries by remember { mutableStateOf<List<ModelRegistryEntry>>(emptyList()) }

    LaunchedEffect(registryService) {
        // Reading the registry is async (asset I/O). Tolerate failure: if
        // the asset is missing we show an empty picker; the wizard still
        // advances with the default id.
        runCatching { registryService.registry().all() }
            .onSuccess { loaded ->
                entries = loaded
                val hasDefault = loaded.any { it.id == "BNLeaky_Keras" }
                selectedModelId = if (hasDefault) "BNLeaky_Keras" else loaded.firstOrNull()?.id ?: "BNLeaky_Keras"
            }
    }

    WizardStepContainer(
        title = "Inference policy",
        subtitle = "These choices apply to every prediction this device makes. Admins can adjust them later from Settings.",
        stepIndicator = "Step 6 of 8",
        primaryLabel = "Continue",
        onPrimary = {
            scope.launch {
                onboarding.setInferencePolicy(
                    threshold = threshold,
                    defaultModelId = selectedModelId,
                    autoLogoutMinutes = autoLogoutMinutes,
                )
            }
        },
    ) {
        ThresholdSlider(threshold = threshold, onThresholdChange = { threshold = it })

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Default model", style = MaterialTheme.typography.titleSmall)
            ModelPicker(
                selectedId = selectedModelId,
                entries = entries,
                onSelectionChange = { selectedModelId = it },
            )
        }

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Auto-logout", style = MaterialTheme.typography.titleSmall)
            val options = listOf(5, 15, 30)
            SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                options.forEachIndexed { index, minutes ->
                    SegmentedButton(
                        selected = autoLogoutMinutes == minutes,
                        onClick = { autoLogoutMinutes = minutes },
                        shape = SegmentedButtonDefaults.itemShape(index = index, count = options.size),
                    ) {
                        Text("$minutes min")
                    }
                }
            }
            Text(
                text = "The app locks itself after this much inactivity. Microscopists re-unlock with biometric.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
