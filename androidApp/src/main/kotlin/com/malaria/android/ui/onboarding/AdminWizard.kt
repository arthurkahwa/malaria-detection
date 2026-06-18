package com.malaria.android.ui.onboarding

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.OnboardingState
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.admin.AdminBiometricStep
import com.malaria.android.ui.onboarding.admin.ClinicDetailsStep
import com.malaria.android.ui.onboarding.admin.DisclaimerAckStep
import com.malaria.android.ui.onboarding.admin.InferencePolicyStep
import com.malaria.android.ui.onboarding.admin.LanguageStep
import com.malaria.android.ui.onboarding.admin.LicenseAckStep
import com.malaria.android.ui.onboarding.admin.ProvisioningCompleteStep
import com.malaria.android.ui.onboarding.admin.WelcomeStep

/**
 * Phase 1 sub-coordinator. Dispatches on [OnboardingState.adminStep]
 * (spec §10) and renders the matching step view. Step views drive
 * transitions by calling `OnboardingState.*` methods.
 *
 * Mirrors `iosApp/Views/Onboarding/AdminWizardView.swift`.
 */
@Composable
fun AdminWizard() {
    val onboarding = LocalOnboardingState.current
    val step by onboarding.adminStep.collectAsStateWithLifecycle()

    when (step) {
        OnboardingState.AdminStep.Language -> LanguageStep()
        OnboardingState.AdminStep.Welcome -> WelcomeStep()
        OnboardingState.AdminStep.LicenseAck -> LicenseAckStep()
        OnboardingState.AdminStep.DisclaimerAck -> DisclaimerAckStep()
        OnboardingState.AdminStep.ClinicDetails -> ClinicDetailsStep()
        OnboardingState.AdminStep.InferencePolicy -> InferencePolicyStep()
        OnboardingState.AdminStep.Biometric -> AdminBiometricStep()
        OnboardingState.AdminStep.ProvisioningComplete -> ProvisioningCompleteStep()
    }
}
