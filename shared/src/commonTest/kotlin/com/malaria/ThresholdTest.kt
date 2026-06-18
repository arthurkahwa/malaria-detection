package com.malaria

import com.malaria.util.Threshold
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ThresholdTest {

    @Test
    fun defaultsHaveExpectedValues() {
        assertEquals(0.3f, Threshold.DEFAULT)
        assertEquals(0.3f, Threshold.GRAY_ZONE_LOW)
        assertEquals(0.7f, Threshold.GRAY_ZONE_HIGH)
    }

    @Test
    fun labelAtZeroIsUninfected() {
        assertEquals("Uninfected", Threshold.label(0.0f))
    }

    @Test
    fun labelAtOneIsParasitized() {
        assertEquals("Parasitized", Threshold.label(1.0f))
    }

    @Test
    fun labelExactlyAtDefaultThresholdIsParasitized() {
        // Boundary: >= threshold => Parasitized.
        assertEquals("Parasitized", Threshold.label(0.3f))
    }

    @Test
    fun labelJustBelowDefaultThresholdIsUninfected() {
        assertEquals("Uninfected", Threshold.label(0.29999f))
    }

    @Test
    fun labelHonorsCustomThreshold() {
        assertEquals("Uninfected", Threshold.label(0.4f, threshold = 0.5f))
        assertEquals("Parasitized", Threshold.label(0.5f, threshold = 0.5f))
    }

    @Test
    fun grayZoneBoundariesAreInclusive() {
        assertTrue(Threshold.shouldFlagForReview(0.3f))
        assertTrue(Threshold.shouldFlagForReview(0.7f))
    }

    @Test
    fun grayZoneInteriorIsFlagged() {
        assertTrue(Threshold.shouldFlagForReview(0.5f))
        assertTrue(Threshold.shouldFlagForReview(0.301f))
        assertTrue(Threshold.shouldFlagForReview(0.699f))
    }

    @Test
    fun outsideGrayZoneIsNotFlagged() {
        assertFalse(Threshold.shouldFlagForReview(0.0f))
        assertFalse(Threshold.shouldFlagForReview(0.29999f))
        assertFalse(Threshold.shouldFlagForReview(0.70001f))
        assertFalse(Threshold.shouldFlagForReview(1.0f))
    }
}
