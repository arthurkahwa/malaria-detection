package com.malaria.android.ui.onboarding.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Decision-threshold slider (spec §10 step 6 and §11 inference policy).
 *
 * Range 0.0–1.0 in 0.05 increments. Shows the live value and the spec's
 * trade-off explanatory caption so the admin understands what they're
 * dialing in.
 *
 * Mirrors `iosApp/Views/Onboarding/Components/ThresholdSlider.swift`.
 */
@Composable
fun ThresholdSlider(
    threshold: Double,
    onThresholdChange: (Double) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                text = "Decision threshold",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "%.2f".format(threshold),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        // Slider expects Float; the underlying onboarding state stores Double.
        Slider(
            value = threshold.toFloat(),
            onValueChange = { onThresholdChange(it.toDouble()) },
            valueRange = 0.0f..1.0f,
            steps = 19, // 0.05 increments between 0 and 1 → 20 stops, 19 inner steps
            modifier = Modifier.fillMaxWidth(),
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text("0.0", style = MaterialTheme.typography.labelSmall)
            Text("1.0", style = MaterialTheme.typography.labelSmall)
        }

        Text(
            text = "Lower values flag more cells as parasitized (more false positives, fewer missed parasites). Higher values flag fewer cells (more missed parasites, fewer false positives). 0.30 is the v1 default.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
