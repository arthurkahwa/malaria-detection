package com.malaria.android.ui.history.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.locals.LocalPredictionStore
import kotlinx.coroutines.launch

/**
 * Free-text session label editor. Spec §13:
 *
 *  - Max 20 characters
 *  - Plain ASCII only (printable 0x20..0x7E); rejects emoji, accented chars,
 *    CJK
 *  - Persistent warning copy at the top about PII in exports
 *
 * Mirrors `iosApp/Views/History/SessionRelabelView.swift`. The validator
 * sits in [SessionRelabelValidator] so the same predicate can be unit
 * tested without spinning up Compose.
 */
@Composable
fun SessionRelabelScreen(sessionId: String, navigator: HistoryNavigator) {
    val store = LocalPredictionStore.current
    val scope = rememberCoroutineScope()
    var label by remember { mutableStateOf("") }

    val canSave = SessionRelabelValidator.isValid(label)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Surface(
            color = Color(0xFFFFF3E0),
            shape = androidx.compose.foundation.shape.RoundedCornerShape(8.dp),
        ) {
            Text(
                text = "Labels appear in exports and audit reports. Do not include " +
                    "identifying information per your clinic's privacy policy.",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF6A4500),
                modifier = Modifier.padding(start = 12.dp, end = 12.dp, top = 10.dp, bottom = 10.dp),
            )
        }

        OutlinedTextField(
            value = label,
            onValueChange = { new ->
                label = if (new.length > SessionRelabelValidator.MAX_LENGTH) {
                    new.take(SessionRelabelValidator.MAX_LENGTH)
                } else {
                    new
                }
            },
            label = { Text("Session label") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "${label.length}/${SessionRelabelValidator.MAX_LENGTH} characters",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f),
            )
            if (label.isNotEmpty() && !SessionRelabelValidator.isValid(label)) {
                Text(
                    text = "ASCII only, no emoji",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Button(
                onClick = {
                    scope.launch {
                        store.relabel(sessionId, label)
                        navigator.pop()
                    }
                },
                enabled = canSave,
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 12.dp),
            ) {
                Text("Save")
            }
            OutlinedButton(
                onClick = { navigator.pop() },
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 12.dp),
            ) {
                Text("Cancel")
            }
        }
    }
}

/**
 * Pure validation predicate. Exposed for unit tests so the rule lives in
 * one place — Compose doesn't compile out of @Composable functions, so
 * the rule has to live outside the screen composable to be JVM-testable.
 *
 * Mirrors `SessionRelabelView.isValidLabel(_:)` on iOS.
 */
object SessionRelabelValidator {
    const val MAX_LENGTH = 20

    fun isValid(label: String): Boolean {
        if (label.isEmpty()) return false
        if (label.length > MAX_LENGTH) return false
        for (c in label) {
            val code = c.code
            if (code < 0x20 || code > 0x7E) return false
        }
        return true
    }
}
