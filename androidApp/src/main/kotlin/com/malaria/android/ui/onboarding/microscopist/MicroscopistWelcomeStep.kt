package com.malaria.android.ui.onboarding.microscopist

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import kotlinx.coroutines.launch

/**
 * Phase 2 step 1 (spec §10): welcome blurb confirming the clinic name this
 * device was provisioned for. "Continue" calls `startMicroscopistClaim()`
 * which advances to the initials step.
 *
 * Mirrors `iosApp/Views/Onboarding/Microscopist/MicroscopistWelcomeStep.swift`.
 */
@Composable
fun MicroscopistWelcomeStep() {
    val onboarding = LocalOnboardingState.current
    val scope = rememberCoroutineScope()
    val pendingName by onboarding.pendingClinicName.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(errorMessage) {
        val message = errorMessage ?: return@LaunchedEffect
        snackbarHostState.showSnackbar(message)
        errorMessage = null
    }

    val clinicName = pendingName ?: "this clinic"

    WizardStepContainer(
        title = "Welcome, microscopist",
        stepIndicator = "Step 1 of 4",
        primaryLabel = "Continue",
        onPrimary = {
            scope.launch {
                runCatching { onboarding.startMicroscopistClaim() }
                    .onFailure { errorMessage = it.message ?: "Unable to start microscopist claim." }
            }
        },
    ) {
        Text(
            text = "This device is provisioned for $clinicName. The next few screens claim the device as yours.",
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            text = "You'll register your own biometric so only you can unlock the app for routine screening.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier.padding(top = 8.dp),
        ) { data -> Snackbar(snackbarData = data) }
    }
}
