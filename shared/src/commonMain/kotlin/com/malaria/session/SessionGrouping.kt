package com.malaria.session

import com.benasher44.uuid.uuid4
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant

/**
 * Implicit 30-minute gap rule that groups consecutive predictions into a
 * shared `sessionId`.
 */
object SessionGrouping {
    const val SESSION_GAP_MINUTES: Long = 30L

    /**
     * Returns the `sessionId` to assign to a new prediction made at [now].
     *
     * - If there is no previous prediction (either timestamp or id is null),
     *   a fresh UUID is generated.
     * - If [now] is at least [SESSION_GAP_MINUTES] minutes after the previous
     *   prediction, a fresh UUID is generated.
     * - Otherwise the previous session is continued.
     */
    fun assignSessionId(
        previousPredictionTimestamp: Instant?,
        previousSessionId: String?,
        now: Instant = Clock.System.now()
    ): String {
        if (previousPredictionTimestamp == null || previousSessionId == null) {
            return uuid4().toString()
        }
        val gap = now - previousPredictionTimestamp
        return if (gap.inWholeMinutes >= SESSION_GAP_MINUTES) {
            uuid4().toString()
        } else {
            previousSessionId
        }
    }
}
