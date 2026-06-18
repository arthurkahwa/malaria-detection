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
import androidx.compose.runtime.mutableIntStateOf
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
import com.malaria.android.services.BiometricPrompter
import com.malaria.android.ui.locals.LocalSettingsStore
import kotlinx.coroutines.launch

/**
 * Edit the auto-logout timeout (admin only). Fresh biometric before
 * `auto_logout_changed` audit entry.
 */
@Composable
fun EditAutoLogoutScreen(navigator: SettingsNavigator) {
    val settings = LocalSettingsStore.current
    val current by settings.autoLogoutMinutes.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selected by remember { mutableIntStateOf(current) }
    var saving by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(current) {
        if (!saving) selected = current
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        listOf(5, 15, 30).forEach { minutes ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .selectable(
                        selected = selected == minutes,
                        onClick = { selected = minutes },
                        role = Role.RadioButton,
                    )
                    .padding(top = 6.dp, bottom = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                RadioButton(selected = selected == minutes, onClick = null)
                Text("$minutes minutes", style = MaterialTheme.typography.bodyMedium)
            }
        }

        Text(
            text = "The app locks itself after this much inactivity. Microscopists re-unlock with biometric.",
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
                        title = "Confirm auto-logout change",
                        subtitle = "Use your fingerprint, face, or device PIN.",
                    )
                    when (outcome) {
                        BiometricPrompter.Outcome.Success -> {
                            settings.updateAutoLogoutMinutes(selected)
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
