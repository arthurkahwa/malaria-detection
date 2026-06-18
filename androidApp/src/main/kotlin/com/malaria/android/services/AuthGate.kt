package com.malaria.android.services

import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.entities.AuditAction
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant

/**
 * Biometric/passcode gate over the device session. Mirrors
 * `iosApp/Services/AuthGate.swift`.
 *
 * Per spec §9: BiometricPrompt (strong biometric or device credential)
 * unlocks the app and holds the session until backgrounded, manually locked,
 * or auto-logged-out (the SessionTimer wiring lands in Phase 7).
 *
 * The biometric prompt itself is presented from a composable via the
 * `onPromptRequired` callback because BiometricPrompt is callback-based and
 * needs a FragmentActivity reference; this service stays Android-API-free
 * so it can be tested with a pure coroutine harness.
 */
class AuthGate(
    private val clinicians: ClinicianDao,
    private val audit: AuditLog,
    private val scope: CoroutineScope,
) {

    sealed interface State {
        data object Locked : State
        data class Unlocked(val sessionStart: Instant) : State
        data object ProvisionedUnclaimed : State
    }

    private val _state = MutableStateFlow<State>(State.Locked)
    val state: StateFlow<State> = _state.asStateFlow()

    /**
     * Refresh initial state from persistence on app launch. Mirrors iOS
     * `AuthGate.bootstrap()`.
     */
    suspend fun bootstrap() {
        val profile = runCatching { clinicians.current() }.getOrNull()
        _state.value = if (profile == null || !profile.biometricEnrolled) {
            State.ProvisionedUnclaimed
        } else {
            State.Locked
        }
    }

    /**
     * Drive a BiometricPrompt via [onPromptRequired]. On success, mark the
     * session unlocked and write the matching audit entries (session_unlocked
     * + auth_success). On failure write auth_failure with the error message.
     */
    suspend fun unlock(onPromptRequired: suspend () -> Result<Unit>) {
        val outcome = runCatching { onPromptRequired() }.getOrElse { Result.failure(it) }
        outcome.onSuccess {
            _state.value = State.Unlocked(Clock.System.now())
            val profile = runCatching { clinicians.current() }.getOrNull()
            val actorId = profile?.actorId ?: "unknown"
            val role = profile?.role ?: "unknown"
            audit.write(AuditAction.SessionUnlocked, actorId = actorId, actorRoleAtTime = role)
            audit.write(AuditAction.AuthSuccess, actorId = actorId, actorRoleAtTime = role)
        }.onFailure { error ->
            val profile = runCatching { clinicians.current() }.getOrNull()
            audit.write(
                AuditAction.AuthFailure,
                actorId = profile?.actorId ?: "unknown",
                actorRoleAtTime = profile?.role ?: "unknown",
                metadata = mapOf("reason" to (error.message ?: "unknown")),
            )
        }
    }

    /**
     * Lock the session and record the matching `session_relocked_*` audit
     * entry. The actor lookup runs in a background coroutine so callers can
     * lock synchronously (e.g. from a foreground-lost lifecycle callback).
     */
    fun lockSession(reason: LockReason) {
        _state.value = State.Locked
        scope.launch {
            val profile = runCatching { clinicians.current() }.getOrNull()
            audit.write(
                action = reason.auditAction,
                actorId = profile?.actorId ?: "unknown",
                actorRoleAtTime = profile?.role ?: "unknown",
            )
        }
    }

    /**
     * Auto-logout via SessionTimer is intentionally disabled for the
     * technology preview. The SessionTimer class exists in shared/ and
     * the setting is configurable in Settings, but the foreground timer
     * is not wired — only manual lock and lifecycle backgrounding fire.
     * Wire SessionTimer.touch() here when auto-logout is enabled.
     */
    fun touch() {
        // Auto-logout intentionally deactivated (technology preview).
    }
}

enum class LockReason {
    Background,
    Manual,
    Timeout;

    internal val auditAction: AuditAction
        get() = when (this) {
            Background -> AuditAction.SessionRelockedBackground
            Manual -> AuditAction.SessionRelockedManual
            Timeout -> AuditAction.SessionRelockedTimeout
        }
}
