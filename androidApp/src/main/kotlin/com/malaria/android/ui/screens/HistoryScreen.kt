package com.malaria.android.ui.screens

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.AuthGate
import com.malaria.android.ui.history.HistoryRoot
import com.malaria.android.ui.locals.LocalAuthGate

/**
 * History tab entry point (spec §11). Gates on [AuthGate]:
 *
 *  - Locked → [LockedPlaceholder]
 *  - ProvisionedUnclaimed → [ProvisioningIncompleteView]
 *  - Unlocked → [HistoryRoot] (Phase 10 subsection + back stack)
 *
 * Mirrors `iosApp/Views/HistoryTab.swift`.
 */
@Composable
fun HistoryScreen() {
    val authGate = LocalAuthGate.current
    val authState by authGate.state.collectAsStateWithLifecycle()

    when (authState) {
        is AuthGate.State.Locked -> LockedPlaceholder()
        is AuthGate.State.Unlocked -> HistoryRoot()
        is AuthGate.State.ProvisionedUnclaimed -> ProvisioningIncompleteView()
    }
}
