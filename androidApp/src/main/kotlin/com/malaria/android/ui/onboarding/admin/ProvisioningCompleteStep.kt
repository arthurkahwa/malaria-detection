package com.malaria.android.ui.onboarding.admin

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.WizardStepContainer

/**
 * Step 8 of admin provisioning (spec §10). The device is now in
 * `provisioned-unclaimed` state. The admin can either hand the device to
 * the microscopist (default path) or — for single-person deployments —
 * continue Phase 2 immediately. Both buttons resolve to the same state
 * transition; the wording differs only for UX clarity.
 *
 * Mirrors `iosApp/Views/Onboarding/Admin/ProvisioningCompleteStep.swift`.
 */
@Composable
fun ProvisioningCompleteStep() {
    val onboarding = LocalOnboardingState.current
    val pendingName by onboarding.pendingClinicName.collectAsStateWithLifecycle()
    val clinicName = pendingName ?: "your clinic"

    WizardStepContainer(
        title = "Device provisioned",
        subtitle = "Phase 1 is complete.",
        stepIndicator = "Step 8 of 8",
        primaryLabel = "Done — hand to microscopist",
        onPrimary = { onboarding.proceedToMicroscopistClaim() },
        secondaryLabel = "You're the microscopist too? Continue Phase 2 now",
        onSecondary = { onboarding.proceedToMicroscopistClaim() },
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(
                    imageVector = Icons.Filled.CheckCircle,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                )
                Text(
                    text = "Device provisioned for $clinicName.",
                    style = MaterialTheme.typography.titleMedium,
                )
            }

            Text(
                text = "Hand this device to the microscopist who will use it. When they open the app they'll be walked through claiming the device and registering their own biometric.",
                style = MaterialTheme.typography.bodyMedium,
            )

            Text(
                text = "Single-person deployment? Tap the alternate button below to continue Phase 2 right now.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
