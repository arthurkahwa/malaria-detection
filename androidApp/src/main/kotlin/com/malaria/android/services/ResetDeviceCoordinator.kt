package com.malaria.android.services

import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.entities.AuditAction

/**
 * Coordinates the spec §10 "Reset device" flow. Mirrors the iOS
 * `ResetDeviceCoordinator` value-for-value.
 *
 * The biometric prompt itself is initiated by the calling composable via
 * [BiometricPrompter] — this coordinator only owns the wipe + audit +
 * onboarding-reset sequence after the prompt resolves.
 */
class ResetDeviceCoordinator(
    private val clinicians: ClinicianDao,
    private val audit: AuditLog,
    private val onboarding: OnboardingState,
    private val settings: SettingsStore,
) {

    /**
     * Execute the wipe. Caller must have already triggered a fresh
     * biometric prompt and double-confirmation per spec §10.
     *
     * Order is significant — identical to iOS:
     *   1. Capture the wiped actor id while the row still exists.
     *   2. Wipe the clinician row (consents preserved per spec §10 "clinic-
     *      level config preserved, clinician-level wiped" interpretation
     *      — only the clinician identity is removed).
     *   3. Write `device_reprovisioned` BEFORE flipping onboarding phase.
     *   4. Reset [OnboardingState] and re-hydrate [SettingsStore].
     */
    suspend fun performReset() {
        val profile = runCatching { clinicians.current() }.getOrNull()
        val wipedActorId = profile?.actorId ?: "unknown"
        val wipedRole = profile?.role ?: "unknown"

        clinicians.wipe()

        audit.write(
            action = AuditAction.DeviceReprovisioned,
            actorId = wipedActorId,
            actorRoleAtTime = wipedRole,
            resourceType = "clinician",
            resourceId = wipedActorId,
            metadata = mapOf("wiped_actor_id" to wipedActorId),
        )

        onboarding.reset()
        settings.hydrate()
    }
}
