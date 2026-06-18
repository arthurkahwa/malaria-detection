package com.malaria.session

import kotlinx.datetime.Clock
import kotlinx.datetime.Instant

/**
 * Auto-logout timer used by the partial-lock model.
 *
 * NOTE: Auto-logout is intentionally deactivated in the technology preview.
 * Neither platform wires [touch] or schedules [checkTimeout] — only manual
 * lock and app-backgrounding fire. This class is retained so the feature can
 * be enabled without an API change: wire AuthGate.touch() → SessionTimer.touch()
 * and schedule checkTimeout() in a platform foreground coroutine to activate.
 */
class SessionTimer(
    private val timeoutMinutes: Int = 30,
    private val onTimeout: suspend () -> Unit
) {
    private var lastActivity: Instant = Clock.System.now()

    /** Resets the inactivity counter to `now`. */
    fun touch() {
        lastActivity = Clock.System.now()
    }

    /** Fires [onTimeout] when inactivity has exceeded [timeoutMinutes]. */
    suspend fun checkTimeout() {
        val elapsed = Clock.System.now() - lastActivity
        if (elapsed.inWholeMinutes >= timeoutMinutes) {
            onTimeout()
        }
    }
}
