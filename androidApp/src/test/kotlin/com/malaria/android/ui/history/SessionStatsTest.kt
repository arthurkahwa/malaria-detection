package com.malaria.android.ui.history

import com.malaria.android.data.entities.Prediction
import com.malaria.android.ui.history.components.RiskBandIndicator
import com.malaria.android.ui.history.components.SessionStats
import com.malaria.android.ui.history.screens.SessionRelabelValidator
import com.malaria.util.Threshold
import kotlinx.datetime.Instant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.UUID

/**
 * Phase 10 history-tab tests. Mirrors `iosApp/Tests/HistoryTabTests.swift`
 * but is JVM-pure (no Compose runtime, no Room) — Compose-side rendering
 * tests live separately on the device once instrumented testing lands.
 *
 * Covers:
 *   - SessionStats.grouped shape and counts
 *   - SessionStats stats (parasitized count, gray-zone count, mean prob)
 *   - RiskBandIndicator classification at 0.2/0.5/0.9 and inclusive
 *     gray-zone boundaries
 *   - SessionRelabelValidator ASCII-only rule (emoji, accented, CJK)
 */
class SessionStatsTest {

    // MARK: - SessionStats.grouped

    @Test
    fun sessionsGrouping_shapeAndCounts() {
        val sessionA = UUID.randomUUID().toString()
        val sessionB = UUID.randomUUID().toString()
        val predictions = listOf(
            samplePrediction(sessionId = sessionA, timestampMs = 1_000_000),
            samplePrediction(sessionId = sessionA, timestampMs = 2_000_000),
            samplePrediction(sessionId = sessionA, timestampMs = 3_000_000),
            samplePrediction(sessionId = sessionB, timestampMs = 5_000_000),
            samplePrediction(sessionId = sessionB, timestampMs = 6_000_000),
        )

        val groups = SessionStats.grouped(predictions)

        assertEquals(2, groups.size)
        val byId = groups.associateBy { it.sessionId }
        assertEquals(3, byId[sessionA]?.count)
        assertEquals(2, byId[sessionB]?.count)
        assertEquals(setOf(sessionA, sessionB), byId.keys)
        // Session B has the latest timestamp → sorted first.
        assertEquals(sessionB, groups.first().sessionId)
    }

    @Test
    fun sessionsGrouping_grayZoneAndParasitizedCounts() {
        val session = UUID.randomUUID().toString()
        // 0.20 → low, 0.50 → gray zone, 0.90 → high, 0.10 → low
        val predictions = listOf(
            samplePrediction(sessionId = session, parasitizedProb = 0.20, label = "Uninfected", threshold = 0.3),
            samplePrediction(sessionId = session, parasitizedProb = 0.50, label = "Parasitized", threshold = 0.3),
            samplePrediction(sessionId = session, parasitizedProb = 0.90, label = "Parasitized", threshold = 0.3),
            samplePrediction(sessionId = session, parasitizedProb = 0.10, label = "Uninfected", threshold = 0.3),
        )

        val stats = SessionStats.from(predictions)
        assertTrue("expected stats", stats != null)
        stats!!
        assertEquals(4, stats.count)
        assertEquals(2, stats.parasitizedCount)
        assertEquals(1, stats.grayZoneCount)
        assertEquals(
            (0.20 + 0.50 + 0.90 + 0.10) / 4.0,
            stats.meanParasitizedProb,
            0.0001,
        )
    }

    // MARK: - RiskBandIndicator

    @Test
    fun riskBand_low_grayZone_high() {
        assertEquals(0.3f, Threshold.GRAY_ZONE_LOW, 0.0001f)
        assertEquals(0.7f, Threshold.GRAY_ZONE_HIGH, 0.0001f)

        assertEquals(RiskBandIndicator.Band.Low, RiskBandIndicator.band(0.20))
        assertEquals(RiskBandIndicator.Band.GrayZone, RiskBandIndicator.band(0.50))
        assertEquals(RiskBandIndicator.Band.High, RiskBandIndicator.band(0.90))

        // Inclusive boundaries on the gray zone.
        assertEquals(
            RiskBandIndicator.Band.GrayZone,
            RiskBandIndicator.band(Threshold.GRAY_ZONE_LOW.toDouble()),
        )
        assertEquals(
            RiskBandIndicator.Band.GrayZone,
            RiskBandIndicator.band(Threshold.GRAY_ZONE_HIGH.toDouble()),
        )
    }

    // MARK: - SessionRelabelValidator

    @Test
    fun relabelValidation_acceptsPlainAscii() {
        assertTrue(SessionRelabelValidator.isValid("morning slide"))
        assertTrue(SessionRelabelValidator.isValid("A"))
        assertTrue(SessionRelabelValidator.isValid("Slide 42 OK"))
        // Em-dash is non-ASCII so this string must be rejected.
        assertFalse(SessionRelabelValidator.isValid("Ward 3 — slide"))
    }

    @Test
    fun relabelValidation_rejectsEmojiAndNonAscii() {
        assertFalse(SessionRelabelValidator.isValid("morning 🩸 slide")) // blood drop emoji
        assertFalse(SessionRelabelValidator.isValid("café")) // é is non-ASCII
        assertFalse(SessionRelabelValidator.isValid("北京")) // CJK 北京
        assertFalse(SessionRelabelValidator.isValid(""))
        assertFalse(SessionRelabelValidator.isValid("x".repeat(21)))
    }

    // MARK: - Fixtures

    private fun samplePrediction(
        sessionId: String = UUID.randomUUID().toString(),
        timestampMs: Long = 0,
        parasitizedProb: Double = 0.85,
        label: String = "Parasitized",
        threshold: Double = 0.3,
        flaggedForReview: Boolean = false,
        clinicianOverride: String? = null,
        duplicateOfId: String? = null,
    ): Prediction = Prediction(
        id = UUID.randomUUID().toString(),
        sessionId = sessionId,
        timestamp = Instant.fromEpochMilliseconds(timestampMs),
        modelId = "BNLeaky_Keras",
        modelVersion = "BNLeaky_Keras",
        parasitizedProb = parasitizedProb,
        uninfectedProb = 1.0 - parasitizedProb,
        label = label,
        threshold = threshold,
        flaggedForReview = flaggedForReview,
        inferenceMs = 42,
        imageHash = "a".repeat(64),
        clinicianOverride = clinicianOverride,
        duplicateOfId = duplicateOfId,
        appVersion = "0.1.0-test",
        osVersion = "test",
    )
}
