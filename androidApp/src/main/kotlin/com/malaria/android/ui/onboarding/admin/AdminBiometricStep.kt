package com.malaria.android.ui.onboarding.admin

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
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
 * Step 7 of admin provisioning (spec §10): admin biometric registration.
 *
 * Launches [BiometricPrompter] which wraps `BiometricPrompt` with
 * BIOMETRIC_STRONG | DEVICE_CREDENTIAL — the OS authoritatively confirms a
 * usable biometric or device PIN/passcode is enrolled. On success,
 * `completeAdminProvisioning()` creates the admin `ClinicianProfile`,
 * marks it `biometricEnrolled`, and advances to
 * [com.malaria.android.services.OnboardingState.AdminStep.ProvisioningComplete].
 *
 * Mirrors `iosApp/Views/Onboarding/Admin/AdminBiometricStep.swift`.
 */
@Composable
fun AdminBiometricStep() {
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
        title = "Register administrator biometric",
        subtitle = "This authorizes admin-only actions like changing the threshold or wiping the device. It is not used for routine screening.",
        stepIndicator = "Step 7 of 8",
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
                    title = "Register administrator biometric",
                    subtitle = "Use your fingerprint, face, or device PIN.",
                )
                when (outcome) {
                    BiometricPrompter.Outcome.Success -> {
                        runCatching { onboarding.completeAdminProvisioning() }
                            .onFailure { errorMessage = it.message ?: "Provisioning failed." }
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
            androidx.compose.foundation.layout.Row(
                verticalAlignment = Alignment.Top,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(Icons.Filled.Face, contentDescription = null)
                Text(
                    text = "Tap the button below. Use the fingerprint sensor, face unlock, or your device PIN when prompted.",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
            androidx.compose.foundation.layout.Row(
                verticalAlignment = Alignment.Top,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(Icons.Filled.Lock, contentDescription = null)
                Text(
                    text = "The biometric is registered with the operating system, not with this app. We never store the fingerprint or face data.",
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
