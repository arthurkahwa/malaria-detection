package com.malaria.android.ui.onboarding.admin

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
import androidx.compose.ui.unit.dp
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.JurisdictionPicker
import com.malaria.android.ui.onboarding.components.LawfulBasisPicker
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import com.malaria.domain.Jurisdiction
import com.malaria.domain.LawfulBasis
import kotlinx.coroutines.launch

/**
 * Step 5 of admin provisioning (spec §10): clinic name + jurisdiction +
 * lawful basis.
 *
 * Mirrors `iosApp/Views/Onboarding/Admin/ClinicDetailsStep.swift`.
 */
@Composable
fun ClinicDetailsStep() {
    val onboarding = LocalOnboardingState.current
    val scope = rememberCoroutineScope()

    var clinicName by remember { mutableStateOf("") }
    var jurisdiction by remember { mutableStateOf(Jurisdiction.OTHER) }
    var lawfulBasis by remember { mutableStateOf(LawfulBasis.HEALTH_PROVISION) }

    val trimmed = clinicName.trim()

    WizardStepContainer(
        title = "Clinic details",
        subtitle = "These appear on exports and in the audit log. You can change them later in Settings (admin biometric required).",
        stepIndicator = "Step 5 of 8",
        primaryLabel = "Continue",
        primaryEnabled = trimmed.isNotEmpty(),
        onPrimary = {
            scope.launch {
                onboarding.configureClinic(
                    name = trimmed,
                    jurisdiction = jurisdiction.canonical,
                    lawfulBasis = lawfulBasis.canonical,
                    version = "v1.0",
                )
            }
        },
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Clinic name", style = MaterialTheme.typography.titleSmall)
            OutlinedTextField(
                value = clinicName,
                onValueChange = { clinicName = it },
                placeholder = { Text("e.g. Kisumu District Health Centre") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
        }

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Jurisdiction", style = MaterialTheme.typography.titleSmall)
            JurisdictionPicker(
                selection = jurisdiction,
                onSelectionChange = { jurisdiction = it },
            )
        }

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Lawful basis", style = MaterialTheme.typography.titleSmall)
            LawfulBasisPicker(
                selection = lawfulBasis,
                onSelectionChange = { lawfulBasis = it },
            )
        }
    }
}
