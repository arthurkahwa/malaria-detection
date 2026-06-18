package com.malaria.util

/**
 * Threshold logic for classifier output.
 *
 * Identical across platforms; clinical-safety code written once.
 */
object Threshold {
    const val DEFAULT: Float = 0.3f
    const val GRAY_ZONE_LOW: Float = 0.3f
    const val GRAY_ZONE_HIGH: Float = 0.7f

    /**
     * Maps a `parasitizedProb` to a binary label using [threshold].
     * Returns the canonical English label `"Parasitized"` or `"Uninfected"`.
     * UI is responsible for localizing the result at render time.
     */
    fun label(parasitizedProb: Float, threshold: Float = DEFAULT): String =
        if (parasitizedProb >= threshold) "Parasitized" else "Uninfected"

    /**
     * True when the probability lies in the ambiguous gray zone (inclusive of
     * both endpoints) and should be surfaced for clinician review.
     */
    fun shouldFlagForReview(parasitizedProb: Float): Boolean =
        parasitizedProb in GRAY_ZONE_LOW..GRAY_ZONE_HIGH
}
