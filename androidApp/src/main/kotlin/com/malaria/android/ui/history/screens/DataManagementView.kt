package com.malaria.android.ui.history.screens

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.fragment.app.FragmentActivity
import com.malaria.android.services.BiometricPrompter
import com.malaria.android.services.LockReason
import com.malaria.android.ui.history.HistoryDestination
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.locals.LocalAuthGate
import com.malaria.android.ui.locals.LocalExportService
import kotlinx.coroutines.launch

/**
 * Data-management screen (spec §11). Three affordances:
 *
 *  - Export all data — Phase 13: fresh biometric prompt then a signed ZIP
 *    bundle is built via the shared [com.malaria.export.ExportBundleBuilder]
 *    and shared via `Intent.ACTION_SEND`.
 *  - Lock device — live.
 *  - Reset device — Phase 11; navigates to [HistoryDestination.ResetDevice].
 *
 * Mirrors `iosApp/Views/History/DataManagementView.swift` button-for-button.
 */
@Composable
fun DataManagementView(navigator: HistoryNavigator? = null) {
    val authGate = LocalAuthGate.current
    val exportService = LocalExportService.current
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var showLockConfirmation by remember { mutableStateOf(false) }
    var isExporting by remember { mutableStateOf(false) }
    var exportError by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Button(
                onClick = {
                    val activity = context as? FragmentActivity
                    if (activity == null) {
                        exportError = "Cannot present the biometric prompt on this activity host."
                        return@Button
                    }
                    isExporting = true
                    exportError = null
                    scope.launch {
                        try {
                            // Spec §9: Export all data is a fresh-auth
                            // action regardless of session state.
                            val outcome = BiometricPrompter(activity).prompt(
                                title = "Authenticate to export",
                                subtitle = "Use your fingerprint, face, or device PIN.",
                            )
                            when (outcome) {
                                BiometricPrompter.Outcome.Success -> {
                                    val file = exportService.generateBundle()
                                    val uri = FileProvider.getUriForFile(
                                        context,
                                        "${context.packageName}.fileprovider",
                                        file,
                                    )
                                    val intent = Intent(Intent.ACTION_SEND).apply {
                                        type = "application/zip"
                                        putExtra(Intent.EXTRA_STREAM, uri)
                                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                    }
                                    activity.startActivity(
                                        Intent.createChooser(intent, "Share export bundle"),
                                    )
                                }
                                is BiometricPrompter.Outcome.Failure -> exportError = outcome.reason
                                BiometricPrompter.Outcome.Cancelled -> {}
                            }
                        } catch (t: Throwable) {
                            exportError = t.message ?: "Export failed."
                        } finally {
                            isExporting = false
                        }
                    }
                },
                enabled = !isExporting,
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 12.dp),
            ) {
                if (isExporting) {
                    CircularProgressIndicator(
                        modifier = Modifier.padding(end = 8.dp),
                        strokeWidth = 2.dp,
                    )
                    Text("Preparing export…")
                } else {
                    Text("Export all data")
                }
            }
            Text(
                text = "Export creates a portable, signed ZIP bundle of every prediction, audit entry, clinician profile, and consent record on this device. Share via Drive, Gmail, or Nearby Share.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            exportError?.let {
                Text(it, color = MaterialTheme.colorScheme.error)
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Button(
                onClick = { showLockConfirmation = true },
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error,
                    contentColor = MaterialTheme.colorScheme.onError,
                ),
            ) {
                Text("Lock device")
            }
            Text(
                text = "Locks the current session immediately. Biometric or device passcode required to unlock.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Button(
                onClick = { navigator?.push(HistoryDestination.ResetDevice) },
                enabled = navigator != null,
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error,
                    contentColor = MaterialTheme.colorScheme.onError,
                ),
            ) {
                Text("Reset device")
            }
            Text(
                text = "Wipes the clinician profile and returns the device to admin provisioning. Predictions and audit history are preserved as chain-of-custody.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }

    if (showLockConfirmation) {
        AlertDialog(
            onDismissRequest = { showLockConfirmation = false },
            title = { Text("Lock device?") },
            text = {
                Text("The next person using this device must re-authenticate with biometrics or the device passcode.")
            },
            confirmButton = {
                TextButton(onClick = {
                    showLockConfirmation = false
                    authGate.lockSession(reason = LockReason.Manual)
                }) {
                    Text("Lock")
                }
            },
            dismissButton = {
                TextButton(onClick = { showLockConfirmation = false }) {
                    Text("Cancel")
                }
            },
        )
    }
}
