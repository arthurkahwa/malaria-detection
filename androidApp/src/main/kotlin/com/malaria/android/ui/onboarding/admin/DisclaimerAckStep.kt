package com.malaria.android.ui.onboarding.admin

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.ConsentCheckbox
import com.malaria.android.ui.onboarding.components.LegalText
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import kotlinx.coroutines.launch

/**
 * Step 4 of admin provisioning (spec §10): medical-device disclaimer
 * acknowledgement (the repo `NOTICE` file).
 *
 * Mirrors `iosApp/Views/Onboarding/Admin/DisclaimerAckStep.swift`.
 */
@Composable
fun DisclaimerAckStep() {
    val onboarding = LocalOnboardingState.current
    val scope = rememberCoroutineScope()

    var hasRead by remember { mutableStateOf(false) }

    WizardStepContainer(
        title = "Medical-device disclaimer",
        subtitle = "Read the full disclaimer, then acknowledge to continue.",
        stepIndicator = "Step 4 of 8",
        primaryLabel = "Continue",
        primaryEnabled = hasRead,
        onPrimary = {
            scope.launch {
                onboarding.acceptMedicalDisclaimer(version = LegalText.DISCLAIMER_VERSION)
            }
        },
    ) {
        Text(
            text = LegalText.DISCLAIMER_BODY,
            style = MaterialTheme.typography.bodyMedium,
        )

        // Spec §16 onboarding disclosure: surfaces the on-device crash log
        // policy here so the user accepts it alongside the medical-device
        // disclaimer.
        Text(
            text = "Diagnostic crash logs",
            style = MaterialTheme.typography.titleSmall,
        )
        Text(
            text = "If the app crashes, a diagnostic log is saved on this device only. Nothing is sent automatically. You can review and share individual logs from Settings.",
            style = MaterialTheme.typography.bodyMedium,
        )

        ConsentCheckbox(
            checked = hasRead,
            onCheckedChange = { hasRead = it },
            label = "I understand this software is not a certified medical device and accept the disclaimer above.",
        )
    }
}
