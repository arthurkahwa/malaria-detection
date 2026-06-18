package com.malaria

import com.malaria.session.SessionGrouping
import kotlinx.datetime.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class SessionGroupingTest {

    private val anchor: Instant = Instant.parse("2026-01-15T12:00:00Z")

    @Test
    fun firstPredictionGeneratesNewSessionWhenBothPreviousNull() {
        val id = SessionGrouping.assignSessionId(
            previousPredictionTimestamp = null,
            previousSessionId = null,
            now = anchor
        )
        assertTrue(id.isNotBlank())
        // Two consecutive "first" calls must yield distinct UUIDs.
        val id2 = SessionGrouping.assignSessionId(null, null, anchor)
        assertNotEquals(id, id2)
    }

    @Test
    fun missingPreviousTimestampGeneratesNewSession() {
        val id = SessionGrouping.assignSessionId(
            previousPredictionTimestamp = null,
            previousSessionId = "stale-session",
            now = anchor
        )
        assertNotEquals("stale-session", id)
        assertTrue(id.isNotBlank())
    }

    @Test
    fun missingPreviousIdGeneratesNewSession() {
        val id = SessionGrouping.assignSessionId(
            previousPredictionTimestamp = anchor,
            previousSessionId = null,
            now = anchor
        )
        assertTrue(id.isNotBlank())
    }

    @Test
    fun withinGapKeepsSameSessionId() {
        val previousTs = anchor
        val now = Instant.parse("2026-01-15T12:29:59Z") // 29m59s after
        val id = SessionGrouping.assignSessionId(
            previousPredictionTimestamp = previousTs,
            previousSessionId = "session-A",
            now = now
        )
        assertEquals("session-A", id)
    }

    @Test
    fun exactlyAtGapBoundaryStartsNewSession() {
        // The rule is `gap.inWholeMinutes >= 30`, so exactly 30 minutes
        // begins a new session.
        val previousTs = anchor
        val now = Instant.parse("2026-01-15T12:30:00Z")
        val id = SessionGrouping.assignSessionId(
            previousPredictionTimestamp = previousTs,
            previousSessionId = "session-A",
            now = now
        )
        assertNotEquals("session-A", id)
        assertTrue(id.isNotBlank())
    }

    @Test
    fun acrossGapStartsNewSession() {
        val previousTs = anchor
        val now = Instant.parse("2026-01-15T13:30:00Z") // 1h30m later
        val id = SessionGrouping.assignSessionId(
            previousPredictionTimestamp = previousTs,
            previousSessionId = "session-A",
            now = now
        )
        assertNotEquals("session-A", id)
        assertTrue(id.isNotBlank())
    }

    @Test
    fun justUnderBoundaryKeepsSession() {
        val previousTs = anchor
        // 29 minutes 59.999s — inWholeMinutes is 29 (truncates toward zero).
        val now = Instant.fromEpochMilliseconds(
            previousTs.toEpochMilliseconds() + (29L * 60_000L) + 59_999L
        )
        val id = SessionGrouping.assignSessionId(
            previousPredictionTimestamp = previousTs,
            previousSessionId = "session-B",
            now = now
        )
        assertEquals("session-B", id)
    }
}
