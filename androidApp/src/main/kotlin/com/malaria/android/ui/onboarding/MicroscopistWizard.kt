package com.malaria.android.ui.onboarding

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.OnboardingState
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.microscopist.InitialsStep
import com.malaria.android.ui.onboarding.microscopist.MicroscopistBiometricStep
import com.malaria.android.ui.onboarding.microscopist.MicroscopistWelcomeStep
import com.malaria.android.ui.onboarding.microscopist.OrientationStep

/**
 * Phase 2 sub-coordinator. Dispatches on
 * [OnboardingState.microscopistStep] (spec §10).
 *
 * Mirrors `iosApp/Views/Onboarding/MicroscopistWizardView.swift`.
 */
@Composable
fun MicroscopistWizard() {
    val onboarding = LocalOnboardingState.current
    val step by onboarding.microscopistStep.collectAsStateWithLifecycle()

    when (step) {
        OnboardingState.MicroscopistStep.Welcome -> MicroscopistWelcomeStep()
        OnboardingState.MicroscopistStep.Initials -> InitialsStep()
        OnboardingState.MicroscopistStep.Biometric -> MicroscopistBiometricStep()
        OnboardingState.MicroscopistStep.Orientation -> OrientationStep()
    }
}
