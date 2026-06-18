package com.malaria.android.ui.override

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import com.malaria.android.data.entities.Prediction
import com.malaria.android.services.BiometricPrompter
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.locals.LocalDatabase
import com.malaria.android.ui.locals.LocalPredictionStore
import com.malaria.domain.OverrideContext
import com.malaria.domain.OverrideReason
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * Single-screen review-override form (spec §12 review override). Mirrors
 * `iosApp/Views/Override/ReviewOverrideView.swift` field-for-field:
 *  - Corrected verdict (Parasitized | Uninfected) as a FilterChip row
 *  - Reason: DropdownMenu over the five canonical [OverrideReason] cases
 *    (avoids the still-experimental `ExposedDropdownMenuBox` we sidestep
 *    elsewhere in Phase 7)
 *  - Override-by initials: defaults to the device clinician's
 *    [com.malaria.android.data.entities.ClinicianProfile.initials], can be
 *    overwritten if a second person is reviewing (spec §12 second-reviewer
 *    workaround)
 *  - Optional notes
 *  - "I have reviewed the full session context" checkbox — required to
 *    enable Save (spec §12)
 *
 * On Save: triggers a fresh biometric prompt via [BiometricPrompter]
 * (spec §9 — review override is in the fresh-auth list), then calls
 * [com.malaria.android.services.PredictionStore.override] which writes
 * the prediction row's override columns plus a single
 * `override_recorded` audit entry with `contextReviewed = true`. On
 * failure or biometric cancellation, surfaces a Snackbar and stays put.
 *
 * Live override (the 2-tap modal during active screening) is deferred to
 * Phase 8 with the camera UI.
 */
@Composable
fun ReviewOverrideScreen(
    predictionId: String,
    navigator: HistoryNavigator,
) {
    val dao = LocalDatabase.current.predictionDao()
    val clinicianDao = LocalDatabase.current.clinicianDao()
    val store = LocalPredictionStore.current
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    val prediction by produceState<Prediction?>(initialValue = null, key1 = predictionId) {
        value = dao.byId(predictionId)
    }
    val clinicianInitials by produceState<String?>(initialValue = null) {
        value = runCatching { clinicianDao.current()?.initials }.getOrNull()
    }

    var state by remember { mutableStateOf(ReviewOverrideUiState()) }
    var initialsLoaded by remember { mutableStateOf(false) }
    LaunchedEffect(clinicianInitials) {
        if (!initialsLoaded) {
            clinicianInitials?.takeIf { it.isNotEmpty() }?.let {
                state = state.copy(initials = it)
            }
            initialsLoaded = true
        }
    }

    val loaded = prediction
    if (loaded == null) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Loading…", style = MaterialTheme.typography.bodyMedium)
        }
        return
    }

    val probabilityPercent = (loaded.parasitizedProb * 100).roundToInt()
    val canSave = ReviewOverrideValidator.canSave(state)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        // -- Header
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = "The model said: ${loaded.label} ($probabilityPercent%)",
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
            )
            Text(
                text = "Captured: ${formatUtc(loaded.timestamp)}, session ${loaded.sessionId.take(8)}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        HorizontalDivider()

        // -- Corrected verdict
        Section(title = "Corrected verdict") {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                ReviewOverrideValidator.Verdict.entries.forEach { verdict ->
                    FilterChip(
                        selected = state.verdict == verdict,
                        onClick = { state = state.copy(verdict = verdict) },
                        label = { Text(verdict.display) },
                    )
                }
            }
        }

        // -- Reason
        Section(title = "Reason") {
            ReasonDropdown(
                selected = state.reason,
                onSelected = { state = state.copy(reason = it) },
            )
        }

        // -- Override by
        Section(title = "Override by") {
            OutlinedTextField(
                value = state.initials,
                onValueChange = { new ->
                    val capped = if (new.length > 2) new.take(2) else new
                    state = state.copy(initials = capped.uppercase())
                },
                label = { Text("Initials") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            Text(
                text = "Defaults to the device clinician's initials. Overwrite if a second " +
                    "reviewer recorded this override.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        // -- Notes
        Section(title = "Notes") {
            OutlinedTextField(
                value = state.notes,
                onValueChange = { state = state.copy(notes = it) },
                label = { Text("Optional notes") },
                minLines = 3,
                maxLines = 5,
                modifier = Modifier.fillMaxWidth(),
            )
        }

        // -- "I have reviewed" checkbox
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Checkbox(
                checked = state.contextReviewed,
                onCheckedChange = { state = state.copy(contextReviewed = it) },
            )
            Text(
                text = "I have reviewed the full session context for this prediction",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f),
            )
        }
        Text(
            text = "Required to enable Save.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        // -- Buttons
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = {
                    val activity = context as? FragmentActivity
                    if (activity == null) {
                        scope.launch {
                            snackbarHostState.showSnackbar(
                                "Biometric prompt unavailable on this host.",
                            )
                        }
                        return@Button
                    }
                    val verdict = state.verdict ?: return@Button
                    val reason = state.reason ?: return@Button
                    state = state.copy(inFlight = true)
                    scope.launch {
                        val prompter = BiometricPrompter(activity)
                        val outcome = prompter.prompt(
                            title = "Confirm review override",
                            subtitle = null,
                        )
                        when (outcome) {
                            BiometricPrompter.Outcome.Success -> {
                                val result = runCatching {
                                    store.override(
                                        prediction = loaded,
                                        verdict = verdict.display,
                                        context = OverrideContext.REVIEW.canonical,
                                        reason = reason.canonical,
                                        notes = state.notes.takeIf { it.isNotEmpty() },
                                        actorInitials = state.initials.takeIf { it.isNotEmpty() },
                                        contextReviewed = true,
                                    )
                                }
                                if (result.isSuccess) {
                                    snackbarHostState.showSnackbar("Override saved.")
                                    navigator.pop()
                                } else {
                                    snackbarHostState.showSnackbar(
                                        result.exceptionOrNull()?.message
                                            ?: "Failed to save override.",
                                    )
                                }
                            }
                            is BiometricPrompter.Outcome.Failure -> {
                                snackbarHostState.showSnackbar(outcome.reason)
                            }
                            BiometricPrompter.Outcome.Cancelled -> {
                                // No-op — let the user retry.
                            }
                        }
                        state = state.copy(inFlight = false)
                    }
                },
                enabled = canSave && !state.inFlight,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (state.inFlight) "Verifying…" else "Save override")
            }
            OutlinedButton(
                onClick = { navigator.pop() },
                enabled = !state.inFlight,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Cancel")
            }
        }

        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier.padding(top = 8.dp),
        ) { data -> Snackbar(snackbarData = data) }
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        content()
    }
}

@Composable
private fun ReasonDropdown(
    selected: OverrideReason?,
    onSelected: (OverrideReason) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier = Modifier.fillMaxWidth()) {
        OutlinedButton(
            onClick = { expanded = true },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(
                text = selected?.let { ReviewOverrideValidator.displayLabel(it) } ?: "Choose…",
                modifier = Modifier.weight(1f),
            )
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            // Spec §15: reason labels stay English even in localised builds —
            // the canonical OverrideReason values map to fixed display
            // strings via ReviewOverrideValidator.displayLabel.
            ReviewOverrideValidator.allReasons.forEach { reason ->
                DropdownMenuItem(
                    text = { Text(ReviewOverrideValidator.displayLabel(reason)) },
                    onClick = {
                        onSelected(reason)
                        expanded = false
                    },
                )
            }
        }
    }
}

private fun formatUtc(instant: kotlinx.datetime.Instant): String {
    val local = instant.toString() // ISO-8601 UTC e.g. 2026-03-14T09:23:45.123Z
    return local
}

/**
 * UI state for the review-override screen. Held as a single value-typed
 * object so the JVM tests can assert the `canSave` rule purely by feeding
 * states through [ReviewOverrideValidator] without spinning up Compose.
 */
data class ReviewOverrideUiState(
    val verdict: ReviewOverrideValidator.Verdict? = null,
    val reason: OverrideReason? = null,
    val initials: String = "",
    val notes: String = "",
    val contextReviewed: Boolean = false,
    val inFlight: Boolean = false,
)

/**
 * Pure validator / label resolver. Mirrors
 * `ReviewOverrideView.canSave(...)` and `ReviewOverrideView.displayLabel(...)`
 * on iOS; lives outside the composable so it's JVM-testable.
 */
object ReviewOverrideValidator {

    enum class Verdict(val display: String) {
        Parasitized("Parasitized"),
        Uninfected("Uninfected"),
    }

    val allReasons: List<OverrideReason> = listOf(
        OverrideReason.IMAGE_QUALITY,
        OverrideReason.ATYPICAL_MORPHOLOGY,
        OverrideReason.MODEL_FALSE_POSITIVE,
        OverrideReason.MODEL_FALSE_NEGATIVE,
        OverrideReason.OTHER,
    )

    fun canSave(state: ReviewOverrideUiState): Boolean =
        state.verdict != null && state.reason != null && state.contextReviewed

    /**
     * Spec §15: override-reason labels stay English even when the rest of
     * the app is localised. Mapping is centralised here so the canonical
     * enum stays the single source of truth.
     */
    fun displayLabel(reason: OverrideReason): String = when (reason) {
        OverrideReason.IMAGE_QUALITY -> "Image quality"
        OverrideReason.ATYPICAL_MORPHOLOGY -> "Atypical morphology"
        OverrideReason.MODEL_FALSE_POSITIVE -> "Model false-positive"
        OverrideReason.MODEL_FALSE_NEGATIVE -> "Model false-negative"
        OverrideReason.OTHER -> "Other"
    }
}
