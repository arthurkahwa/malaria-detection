package com.malaria.android.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Top-level Settings screen and back-stack host. Mirrors the iOS
 * `SettingsTab.swift` NavigationStack. Sections render in spec-prescribed
 * order on the root destination; editable rows push their dedicated
 * screen onto [SettingsNavigator].
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsRoot() {
    val navigator = rememberSettingsNavigator()
    val current = navigator.current

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(currentTitle(current)) },
                navigationIcon = {
                    if (navigator.canGoBack) {
                        IconButton(onClick = { navigator.pop() }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back",
                            )
                        }
                    }
                },
            )
        },
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            when (val destination = current) {
                SettingsDestination.Root -> SettingsList(navigator)
                SettingsDestination.EditInitials -> EditInitialsScreen(navigator)
                SettingsDestination.EditThreshold -> EditThresholdScreen(navigator)
                SettingsDestination.EditDefaultModel -> EditDefaultModelScreen(navigator)
                SettingsDestination.EditAutoLogout -> EditAutoLogoutScreen(navigator)
                SettingsDestination.EditLanguage -> EditLanguageScreen(navigator)
                is SettingsDestination.LegalDocument -> LegalDocumentScreen(
                    title = destination.title,
                    body = destination.body,
                )
                SettingsDestination.ResetDevice -> ResetDeviceScreen(navigator)
                SettingsDestination.CrashLogs -> CrashLogsScreen()
            }
        }
    }
}

@Composable
private fun SettingsList(navigator: SettingsNavigator) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 12.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        item { ClinicSection() }
        item { ClinicianSection(navigator) }
        item { InferenceSection(navigator) }
        item { ModelsSection() }
        item { LanguageSection(navigator) }
        item { LegalSection(navigator) }
        item { CrashLogsSection(navigator) }
    }
}

private fun currentTitle(destination: SettingsDestination): String = when (destination) {
    SettingsDestination.Root -> "Settings"
    SettingsDestination.EditInitials -> "Edit initials"
    SettingsDestination.EditThreshold -> "Edit threshold"
    SettingsDestination.EditDefaultModel -> "Default model"
    SettingsDestination.EditAutoLogout -> "Auto-logout"
    SettingsDestination.EditLanguage -> "Language"
    is SettingsDestination.LegalDocument -> destination.title
    SettingsDestination.ResetDevice -> "Reset device"
    SettingsDestination.CrashLogs -> "Crash logs"
}
