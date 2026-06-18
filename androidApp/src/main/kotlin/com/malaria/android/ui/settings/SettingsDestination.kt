package com.malaria.android.ui.settings

/**
 * In-memory navigation state for the Settings tab. Same in-house pattern
 * as [com.malaria.android.ui.history.HistoryDestination] — no Jetpack
 * Navigation dependency added in Phase 11.
 *
 * Mirrors the iOS NavigationStack pushes inside `SettingsTab.swift` —
 * every `NavigationLink` becomes a `Destination` here.
 */
sealed interface SettingsDestination {

    /** Sections list. */
    data object Root : SettingsDestination

    data object EditInitials : SettingsDestination

    data object EditThreshold : SettingsDestination

    data object EditDefaultModel : SettingsDestination

    data object EditAutoLogout : SettingsDestination

    data object EditLanguage : SettingsDestination

    data class LegalDocument(val title: String, val body: String) : SettingsDestination

    data object ResetDevice : SettingsDestination

    /** Phase 14 — crash log list (spec §16). */
    data object CrashLogs : SettingsDestination
}
