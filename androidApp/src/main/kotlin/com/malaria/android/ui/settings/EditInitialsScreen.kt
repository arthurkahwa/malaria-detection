package com.malaria.android.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import com.malaria.android.services.BiometricPrompter
import com.malaria.android.ui.locals.LocalDatabase
import com.malaria.android.ui.locals.LocalSettingsStore
import kotlinx.coroutines.launch

/**
 * Edit microscopist initials. Triggers a fresh biometric prompt before
 * writing per spec §9.
 */
@Composable
fun EditInitialsScreen(navigator: SettingsNavigator) {
    val settings = LocalSettingsStore.current
    val database = LocalDatabase.current
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var initials by remember { mutableStateOf("") }
    var saving by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        initials = runCatching { database.clinicianDao().current()?.initials }.getOrNull() ?: ""
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        OutlinedTextField(
            value = initials,
            onValueChange = { new ->
                initials = if (new.length > 2) new.take(2) else new
            },
            label = { Text("Initials") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Text(
            text = "Up to two characters. Appears next to overrides in the audit log.",
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
                        title = "Confirm initials change",
                        subtitle = "Use your fingerprint, face, or device PIN.",
                    )
                    when (outcome) {
                        BiometricPrompter.Outcome.Success -> {
                            val trimmed = initials.trim()
                            settings.updateInitials(trimmed.ifEmpty { null })
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

        errorText?.let {
            Text(it, color = MaterialTheme.colorScheme.error)
        }
    }
}
