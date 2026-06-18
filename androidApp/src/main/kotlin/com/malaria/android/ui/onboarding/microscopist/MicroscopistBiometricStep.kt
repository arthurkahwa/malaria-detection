package com.malaria.android.ui.onboarding.microscopist

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.Icon
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import com.malaria.android.services.BiometricPrompter
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import kotlinx.coroutines.launch

/**
 * Phase 2 step 3 (spec §10): microscopist biometric registration. Mirrors
 * [com.malaria.android.ui.onboarding.admin.AdminBiometricStep] but for the
 * routine-unlock biometric rather than the admin authorisation biometric.
 *
 * Mirrors `iosApp/Views/Onboarding/Microscopist/MicroscopistBiometricStep.swift`.
 */
@Composable
fun MicroscopistBiometricStep() {
    val onboarding = LocalOnboardingState.current
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    var inFlight by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(errorMessage) {
        val message = errorMessage ?: return@LaunchedEffect
        snackbarHostState.showSnackbar(message)
        errorMessage = null
    }

    WizardStepContainer(
        title = "Register your biometric",
        subtitle = "Use fingerprint, face unlock, or your device PIN. This is how you'll unlock the app each session.",
        stepIndicator = "Step 3 of 4",
        primaryLabel = if (inFlight) "Verifying…" else "Register biometric",
        primaryEnabled = !inFlight,
        onPrimary = {
            val activity = context as? FragmentActivity
                ?: run {
                    errorMessage = "Biometric registration unavailable on this activity host."
                    return@WizardStepContainer
                }
            inFlight = true
            scope.launch {
                val prompter = BiometricPrompter(activity)
                val outcome = prompter.prompt(
                    title = "Register microscopist biometric",
                    subtitle = "Use your fingerprint, face, or device PIN.",
                )
                when (outcome) {
                    BiometricPrompter.Outcome.Success -> {
                        runCatching { onboarding.completeMicroscopistClaim() }
                            .onFailure { errorMessage = it.message ?: "Could not complete claim." }
                    }
                    is BiometricPrompter.Outcome.Failure -> {
                        errorMessage = outcome.reason
                    }
                    BiometricPrompter.Outcome.Cancelled -> {
                        // No-op — let the user try again.
                    }
                }
                inFlight = false
            }
        },
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                verticalAlignment = Alignment.Top,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(Icons.Filled.Lock, contentDescription = null)
                Text(
                    text = "The biometric is registered with the operating system. The app never sees your face or fingerprint data.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(
                verticalAlignment = Alignment.Top,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(Icons.Filled.CheckCircle, contentDescription = null)
                Text(
                    text = "After this step you'll see a short orientation, then the app is ready for screening.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier.padding(top = 8.dp),
        ) { data -> Snackbar(snackbarData = data) }
    }
}
