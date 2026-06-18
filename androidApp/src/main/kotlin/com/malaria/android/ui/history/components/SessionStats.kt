package com.malaria.android.ui.history.components

import com.malaria.android.data.entities.Prediction
import com.malaria.util.Threshold
import kotlinx.datetime.Instant
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime
import kotlin.math.roundToInt

/**
 * Aggregated stats for one session, computed in Kotlin from a list of
 * [Prediction]. Mirrors `iosApp/Views/History/Components/SessionRowView.swift`'s
 * `SessionStats` value type bit-for-bit (spec §13).
 *
 * Stats live client-side in Phase 10; the v1.1 plan moves the aggregator
 * into shared Kotlin per spec §13. Putting it in the Android `ui/history`
 * package now means callers don't need to round-trip through a repo for a
 * pure-function computation.
 */
data class SessionStats(
    val sessionId: String,
    val sessionLabel: String?,
    val count: Int,
    val parasitizedCount: Int,
    val grayZoneCount: Int,
    val meanParasitizedProb: Double,
    val earliest: Instant,
    val latest: Instant,
) {

    /** First-line title; falls back to a session-id prefix when unlabeled. */
    val displayLabel: String
        get() = sessionLabel?.takeIf { it.isNotEmpty() }
            ?: "Session ${sessionId.take(8)}"

    val meanParasitizedFormatted: String
        get() = "${(meanParasitizedProb * 100).roundToInt()}%"

    /**
     * Compact date-range label. Same-day → "May 15, 2026 · 09:00–11:30";
     * cross-day → "5/15/26 · 09:00 – 5/17/26 · 11:30".
     *
     * Uses `kotlinx-datetime` LocalDateTime conversion via the system zone
     * — keeps the format string compact and avoids pulling in Java
     * `DateFormatter` here. Matches iOS output where the user is in the
     * device's local timezone (the iOS view also formats in local time).
     */
    val dateRangeLabel: String
        get() {
            val zone = TimeZone.currentSystemDefault()
            val first = earliest.toLocalDateTime(zone)
            val last = latest.toLocalDateTime(zone)
            val firstDate = "${first.year}-${first.monthNumber.pad()}-${first.dayOfMonth.pad()}"
            val lastDate = "${last.year}-${last.monthNumber.pad()}-${last.dayOfMonth.pad()}"
            val firstTime = "${first.hour.pad()}:${first.minute.pad()}"
            val lastTime = "${last.hour.pad()}:${last.minute.pad()}"
            return if (firstDate == lastDate) {
                "$firstDate · $firstTime–$lastTime"
            } else {
                "$firstDate $firstTime – $lastDate $lastTime"
            }
        }

    companion object {

        /**
         * Build stats from a flat list of predictions, all assumed to share a
         * `sessionId`. Returns null on empty input — mirrors the optional
         * return on iOS so empty-session sections can render a placeholder.
         */
        fun from(predictions: List<Prediction>): SessionStats? {
            if (predictions.isEmpty()) return null
            val first = predictions.first()
            val sortedByTime = predictions.sortedBy { it.timestamp }
            val count = predictions.size
            val parasitizedCount = predictions.count { it.label == "Parasitized" }
            val low = Threshold.GRAY_ZONE_LOW.toDouble()
            val high = Threshold.GRAY_ZONE_HIGH.toDouble()
            val grayZoneCount = predictions.count {
                it.parasitizedProb in low..high
            }
            val meanProb = predictions.sumOf { it.parasitizedProb } / count.toDouble()
            return SessionStats(
                sessionId = first.sessionId,
                sessionLabel = first.sessionLabel,
                count = count,
                parasitizedCount = parasitizedCount,
                grayZoneCount = grayZoneCount,
                meanParasitizedProb = meanProb,
                earliest = sortedByTime.first().timestamp,
                latest = sortedByTime.last().timestamp,
            )
        }

        /**
         * Group a flat prediction list into per-session stats, newest first
         * by latest-timestamp. Mirrors `SessionStats.grouped` on iOS.
         */
        fun grouped(predictions: List<Prediction>): List<SessionStats> =
            predictions.groupBy { it.sessionId }
                .values
                .mapNotNull { from(it) }
                .sortedByDescending { it.latest }
    }
}

private fun Int.pad(): String = toString().padStart(2, '0')
