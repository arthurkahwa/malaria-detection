package com.malaria.android.ui.onboarding.admin

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.WizardStepContainer

/**
 * Step 2 of admin provisioning (spec §10): welcome blurb.
 *
 * Mirrors `iosApp/Views/Onboarding/Admin/WelcomeStep.swift`.
 */
@Composable
fun WelcomeStep() {
    val onboarding = LocalOnboardingState.current

    WizardStepContainer(
        title = "Welcome",
        stepIndicator = "Step 2 of 8",
        primaryLabel = "Continue",
        onPrimary = { onboarding.advanceFromWelcome() },
    ) {
        Text(
            text = "Malaria Detector is a decision-support tool for trained microscopists screening Giemsa-stained thin blood smears for Plasmodium parasites. It runs entirely on this device — no images leave the clinic.",
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            text = "Setup happens in two phases. You (the clinic administrator) configure the device first, then hand it to a microscopist who claims it as their own.",
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            text = "This app does not replace a trained microscopist. It produces a probability score; the microscopist decides.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
