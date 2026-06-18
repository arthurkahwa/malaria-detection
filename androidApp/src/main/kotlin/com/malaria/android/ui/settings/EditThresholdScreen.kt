package com.malaria.android.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.BiometricPrompter
import com.malaria.android.ui.locals.LocalSettingsStore
import kotlinx.coroutines.launch

/**
 * Edit the decision threshold (admin only). Fresh biometric before
 * `threshold_changed` audit entry per spec §9 + §11.
 */
@Composable
fun EditThresholdScreen(navigator: SettingsNavigator) {
    val settings = LocalSettingsStore.current
    val current by settings.threshold.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var threshold by remember { mutableFloatStateOf(current.toFloat()) }
    var saving by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(current) {
        if (!saving) threshold = current.toFloat()
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Slider(
            value = threshold,
            onValueChange = { threshold = it },
            valueRange = 0f..1f,
        )
        Text(
            text = "%.2f".format(threshold),
            style = MaterialTheme.typography.titleMedium,
        )
        Text(
            text = "Lower threshold → more false positives, fewer false negatives. Higher threshold → fewer false positives, more false negatives.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Button(
            onClick = {
                val activity = context as? FragmentActivity
                if (activity == null) {
                    errorText = "Biometric prompt unavailable on this activity host."
                    return@Button
                }
                saving = true
                errorText = null
                scope.launch {
                    val outcome = BiometricPrompter(activity).prompt(
                        title = "Confirm threshold change",
                        subtitle = "Use your fingerprint, face, or device PIN.",
                    )
                    when (outcome) {
                        BiometricPrompter.Outcome.Success -> {
                            settings.updateThreshold(threshold.toDouble())
                            navigator.pop()
                        }
                        is BiometricPrompter.Outcome.Failure -> errorText = outcome.reason
                        BiometricPrompter.Outcome.Cancelled -> {}
                    }
                    saving = false
                }
            },
            enabled = !saving,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (saving) "Saving…" else "Save")
        }
        errorText?.let { Text(it, color = MaterialTheme.colorScheme.error) }
    }
}
