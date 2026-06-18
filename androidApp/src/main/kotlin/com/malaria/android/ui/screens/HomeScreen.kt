package com.malaria.android.ui.screens

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.AuthGate
import com.malaria.android.ui.locals.LocalAuthGate
import com.malaria.android.ui.locals.LocalPredictionStore

/**
 * Per spec §19. Direct service consumption — no ViewModel intermediate.
 */
@Composable
fun HomeScreen() {
    val authGate = LocalAuthGate.current
    val predictionStore = LocalPredictionStore.current

    val authState by authGate.state.collectAsStateWithLifecycle()
    val recentPredictions by predictionStore.recent.collectAsStateWithLifecycle()

    when (authState) {
        is AuthGate.State.Locked -> LockedPlaceholder()
        is AuthGate.State.Unlocked -> ActiveScreeningView(
            recentPrediction = recentPredictions.firstOrNull(),
        )
        is AuthGate.State.ProvisionedUnclaimed -> ProvisioningIncompleteView()
    }
}
