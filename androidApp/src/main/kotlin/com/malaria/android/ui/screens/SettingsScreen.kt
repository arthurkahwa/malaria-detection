package com.malaria.android.ui.screens

import androidx.compose.runtime.Composable
import com.malaria.android.ui.settings.SettingsRoot

/**
 * Settings tab entry point (spec §11). Delegates to [SettingsRoot] which
 * owns the in-house back-stack navigation across the section + edit
 * screens. The tab is always accessible (no auth gate); each editable row
 * gates on a fresh biometric prompt internally.
 */
@Composable
fun SettingsScreen() {
    SettingsRoot()
}
