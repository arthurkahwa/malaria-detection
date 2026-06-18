package com.malaria.android.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.data.OnboardingLanguage
import com.malaria.android.services.BiometricPrompter
import com.malaria.android.ui.locals.LocalSettingsStore
import kotlinx.coroutines.launch

/**
 * Edit the UI language. Fresh biometric prompt required to prevent
 * stranger-flips per spec §11.
 */
@Composable
fun EditLanguageScreen(navigator: SettingsNavigator) {
    val settings = LocalSettingsStore.current
    val currentRaw by settings.language.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selected by remember { mutableStateOf(OnboardingLanguage.fromCanonical(currentRaw)) }
    var saving by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(currentRaw) {
        if (!saving) selected = OnboardingLanguage.fromCanonical(currentRaw)
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        OnboardingLanguage.values().forEach { lang ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .selectable(
                        selected = selected == lang,
                        onClick = { selected = lang },
                        role = Role.RadioButton,
                    )
                    .padding(top = 6.dp, bottom = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                RadioButton(selected = selected == lang, onClick = null)
                Text(lang.displayName, style = MaterialTheme.typography.bodyMedium)
            }
        }

        Text(
            text = "Only English is fully translated in v1. Other locales fall back to English strings.",
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
                        title = "Confirm language change",
                        subtitle = "Use your fingerprint, face, or device PIN.",
                    )
                    when (outcome) {
                        BiometricPrompter.Outcome.Success -> {
                            settings.updateLanguage(selected)
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
