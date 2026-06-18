package com.malaria.android.ui.history.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.malaria.util.Threshold

/**
 * Visual risk-band indicator for a parasitized probability.
 *
 * Mirrors `iosApp/Views/History/Components/RiskBandIndicator.swift`. Bands
 * defined per spec §11 from `Threshold.GRAY_ZONE_LOW`/`HIGH`:
 *
 *   - low  : prob < GRAY_ZONE_LOW (0.3) → gray
 *   - gray : prob ∈ [LOW, HIGH] inclusive → amber
 *   - high : prob > GRAY_ZONE_HIGH (0.7) → red
 *
 * Boundary inclusivity matches the iOS implementation and the
 * `SessionStats.grayZoneCount` predicate so the count and the per-row
 * indicator never disagree.
 */
object RiskBandIndicator {

    enum class Band(val displayLabel: String, val color: Color) {
        Low("Low", Color.Gray),
        GrayZone("Gray zone", Color(0xFFE08C00)),
        High("High", Color(0xFFD32F2F)),
    }

    fun band(parasitizedProb: Double): Band {
        val low = Threshold.GRAY_ZONE_LOW.toDouble()
        val high = Threshold.GRAY_ZONE_HIGH.toDouble()
        return when {
            parasitizedProb < low -> Band.Low
            parasitizedProb > high -> Band.High
            else -> Band.GrayZone
        }
    }
}

@Composable
fun RiskBandIndicator(parasitizedProb: Double, modifier: Modifier = Modifier) {
    val band = RiskBandIndicator.band(parasitizedProb)
    Row(
        modifier = modifier.semantics {
            contentDescription = "Risk band: ${band.displayLabel}"
        },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .background(band.color, CircleShape),
        )
        Text(
            text = band.displayLabel,
            style = MaterialTheme.typography.labelMedium,
            color = band.color,
        )
    }
}
