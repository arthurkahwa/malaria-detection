package com.malaria.android.ui.onboarding

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.OnboardingState
import com.malaria.android.ui.locals.LocalOnboardingState

/**
 * Top-level onboarding coordinator (spec §10).
 *
 * Reads [OnboardingState.phase] and routes to the matching sub-flow:
 * admin provisioning (Phase 1), microscopist claim (Phase 2), or — when
 * phase is [OnboardingState.Phase.Complete] — renders nothing so the
 * composition root shows `RootScreen` instead.
 *
 * Mirrors `iosApp/Views/Onboarding/OnboardingFlow.swift`.
 */
@Composable
fun OnboardingFlow() {
    val onboarding = LocalOnboardingState.current
    val phase by onboarding.phase.collectAsStateWithLifecycle()

    when (phase) {
        OnboardingState.Phase.AdminProvisioning -> AdminWizard()
        OnboardingState.Phase.MicroscopistClaim -> MicroscopistWizard()
        OnboardingState.Phase.Complete -> {
            // Belt-and-braces: the composition root should never mount
            // OnboardingFlow when phase is Complete, but an empty branch
            // keeps the app navigable rather than crashing on a future
            // gating regression.
        }
    }
}
