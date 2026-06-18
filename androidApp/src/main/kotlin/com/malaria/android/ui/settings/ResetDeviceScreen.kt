package com.malaria.android.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
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
import com.malaria.android.ui.locals.LocalResetDeviceCoordinator
import kotlinx.coroutines.launch

/**
 * Spec §10 re-onboarding flow. Reachable from Settings → Reset device,
 * which itself is reachable from History → Data management. Mirrors the
 * iOS `ResetDeviceView`:
 *
 *   1. Show explanatory copy + a destructive primary button.
 *   2. On tap: fresh biometric prompt via [BiometricPrompter].
 *   3. On success: double-confirmation dialog.
 *   4. On final confirm: [ResetDeviceCoordinator.performReset] wipes the
 *      clinician row, writes `device_reprovisioned`, resets
 *      `OnboardingState.phase` back to `.AdminProvisioning`. `MainActivity`
 *      auto-mounts `OnboardingFlow` on next composition.
 */
@Composable
fun ResetDeviceScreen(@Suppress("UNUSED_PARAMETER") navigator: SettingsNavigator) {
    val coordinator = LocalResetDeviceCoordinator.current
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var authenticating by remember { mutableStateOf(false) }
    var showDoubleConfirm by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.error,
            )
            Text(
                text = "This action cannot be undone",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.error,
            )
        }
        Text(
            text = "Resetting the device wipes the clinician profile on this device, then returns the app to admin provisioning. Predictions and audit history are preserved as chain-of-custody.",
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            text = "Phase 1 (admin provisioning) must be completed again on the next launch.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Button(
            onClick = {
                val activity = context as? FragmentActivity
                if (activity == null) {
                    errorText = "Biometric prompt unavailable on this activity host."
                    return@Button
                }
                authenticating = true
                errorText = null
                scope.launch {
                    val outcome = BiometricPrompter(activity).prompt(
                        title = "Confirm device reset",
                        subtitle = "Use your fingerprint, face, or device PIN.",
                    )
                    when (outcome) {
                        BiometricPrompter.Outcome.Success -> showDoubleConfirm = true
                        is BiometricPrompter.Outcome.Failure -> errorText = outcome.reason
                        BiometricPrompter.Outcome.Cancelled -> {}
                    }
                    authenticating = false
                }
            },
            enabled = !authenticating,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.error,
                contentColor = MaterialTheme.colorScheme.onError,
            ),
        ) {
            Text(if (authenticating) "Verifying…" else "Reset device")
        }

        errorText?.let { Text(it, color = MaterialTheme.colorScheme.error) }
    }

    if (showDoubleConfirm) {
        AlertDialog(
            onDismissRequest = { showDoubleConfirm = false },
            title = { Text("Wipe clinician data?") },
            text = {
                Text("This will wipe clinician data on this device. Predictions and audit history are preserved.")
            },
            confirmButton = {
                TextButton(onClick = {
                    showDoubleConfirm = false
                    scope.launch {
                        runCatching { coordinator.performReset() }
                            .onFailure { errorText = it.message ?: "Reset failed." }
                        // The composition root will re-render with
                        // OnboardingFlow as soon as phase flips; this
                        // screen disappears with the History stack.
                    }
                }) {
                    Text("Wipe and re-provision")
                }
            },
            dismissButton = {
                TextButton(onClick = { showDoubleConfirm = false }) {
                    Text("Cancel")
                }
            },
        )
    }
}
