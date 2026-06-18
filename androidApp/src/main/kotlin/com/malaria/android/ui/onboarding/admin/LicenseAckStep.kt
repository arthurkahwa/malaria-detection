package com.malaria.android.ui.onboarding.admin

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.ConsentCheckbox
import com.malaria.android.ui.onboarding.components.LegalText
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import kotlinx.coroutines.launch

/**
 * Step 3 of admin provisioning (spec §10): Hippocratic License
 * acknowledgement.
 *
 * Mirrors `iosApp/Views/Onboarding/Admin/LicenseAckStep.swift`.
 */
@Composable
fun LicenseAckStep() {
    val context = LocalContext.current
    val onboarding = LocalOnboardingState.current
    val scope = rememberCoroutineScope()

    var hasRead by remember { mutableStateOf(false) }

    WizardStepContainer(
        title = "Hippocratic License",
        subtitle = "Read the full license, then acknowledge to continue.",
        stepIndicator = "Step 3 of 8",
        primaryLabel = "Continue",
        primaryEnabled = hasRead,
        onPrimary = {
            scope.launch {
                onboarding.acceptHippocraticLicense(version = LegalText.LICENSE_VERSION)
            }
        },
    ) {
        Text(
            text = LegalText.licenseBody(context),
            style = MaterialTheme.typography.bodyMedium,
        )

        ConsentCheckbox(
            checked = hasRead,
            onCheckedChange = { hasRead = it },
            label = "I have read and accept the Hippocratic License 3.0 (HL3-FULL).",
        )
    }
}
