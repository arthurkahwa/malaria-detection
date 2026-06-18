package com.malaria.android.ui.override

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.malaria.android.data.entities.Prediction
import com.malaria.android.services.PredictionStore
import com.malaria.domain.OverrideContext
import com.malaria.domain.OverrideReason
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * Two-tap live-override modal during active screening (spec §12 live
 * override). Mirror of `iosApp/Views/Override/LiveOverrideSheet.swift`.
 *
 * Screen 1 picks the corrected verdict (Parasitized | Uninfected).
 * Screen 2 picks the canonical [OverrideReason]. Tapping a reason
 * writes the override via [PredictionStore.override] with
 * `context = "live"`, no biometric prompt, no notes, no initials,
 * `contextReviewed = null` — minimum friction during active
 * screening per spec §12 (the review-override flow in History is
 * the path for fuller context).
 *
 * Round-trip target ~3 s. The microscopist returns to the camera
 * ready for the next cell.
 *
 * Uses Material 3 [ModalBottomSheet] — stable in the project's
 * Compose BOM `2026.04.01`. Distinct from the Phase 9 Review
 * override which lives at full-screen depth via [HistoryNavigator].
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LiveOverrideSheet(
    prediction: Prediction,
    onDismiss: () -> Unit,
    predictionStore: PredictionStore,
) {
    val sheetState = rememberModalBottomSheetState(
        skipPartiallyExpanded = true,
    )
    var step by remember { mutableStateOf<LiveOverrideStep>(LiveOverrideStep.Verdict) }
    var errorText by remember { mutableStateOf<String?>(null) }
    var inFlight by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(start = 24.dp, end = 24.dp, top = 8.dp, bottom = 32.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            when (val s = step) {
                LiveOverrideStep.Verdict -> VerdictPicker(
                    prediction = prediction,
                    onPick = { verdict ->
                        errorText = null
                        step = LiveOverrideStep.Reason(verdict)
                    },
                    errorText = errorText,
                )
                is LiveOverrideStep.Reason -> ReasonPicker(
                    verdict = s.verdict,
                    inFlight = inFlight,
                    errorText = errorText,
                    onReason = { reason ->
                        inFlight = true
                        scope.launch {
                            val result = runCatching {
                                predictionStore.override(
                                    prediction = prediction,
                                    verdict = s.verdict.display,
                                    context = OverrideContext.LIVE.canonical,
                                    reason = reason.canonical,
                                    notes = null,
                                    actorInitials = null,
                                    contextReviewed = null,
                                )
                            }
                            inFlight = false
                            if (result.isSuccess) {
                                onDismiss()
                            } else {
                                errorText = result.exceptionOrNull()?.message
                                    ?: "Failed to save override."
                            }
                        }
                    },
                    onBack = {
                        errorText = null
                        step = LiveOverrideStep.Verdict
                    },
                )
            }
        }
    }
}

/**
 * Sealed step type pulled out so the two-screen state is easy to
 * reason about — and so the JVM test suite can pin the canonical
 * strings independently of Compose if needed.
 */
sealed interface LiveOverrideStep {
    data object Verdict : LiveOverrideStep
    data class Reason(val verdict: LiveOverrideVerdict) : LiveOverrideStep
}

enum class LiveOverrideVerdict(val display: String) {
    Parasitized("Parasitized"),
    Uninfected("Uninfected"),
}

@Composable
private fun VerdictPicker(
    prediction: Prediction,
    onPick: (LiveOverrideVerdict) -> Unit,
    errorText: String?,
) {
    val probabilityPercent = (prediction.parasitizedProb * 100).roundToInt()
    Text(
        text = "The model said: ${prediction.label} ($probabilityPercent%)",
        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
    )
    Text(
        text = "Override to:",
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        LiveOverrideVerdict.entries.forEach { verdict ->
            Button(
                onClick = { onPick(verdict) },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(verdict.display)
            }
        }
    }
    if (errorText != null) {
        Text(
            text = errorText,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.error,
        )
    }
    Spacer(modifier = Modifier.padding(top = 8.dp))
}

@Composable
private fun ReasonPicker(
    verdict: LiveOverrideVerdict,
    inFlight: Boolean,
    errorText: String?,
    onReason: (OverrideReason) -> Unit,
    onBack: () -> Unit,
) {
    Text(
        text = "Reason:",
        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
    )
    Text(
        text = "Verdict: ${verdict.display}",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // Spec §15 keeps these labels English. Reuses Phase 9's
        // central [ReviewOverrideValidator.displayLabel] so any
        // future label changes flow to both override surfaces.
        ReviewOverrideValidator.allReasons.forEach { reason ->
            OutlinedButton(
                onClick = { onReason(reason) },
                enabled = !inFlight,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Row(modifier = Modifier.fillMaxWidth()) {
                    Text(ReviewOverrideValidator.displayLabel(reason))
                }
            }
        }
    }
    if (errorText != null) {
        Text(
            text = errorText,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.error,
        )
    }
    OutlinedButton(
        onClick = onBack,
        enabled = !inFlight,
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp),
    ) {
        Text(if (inFlight) "Saving…" else "Back")
    }
}
