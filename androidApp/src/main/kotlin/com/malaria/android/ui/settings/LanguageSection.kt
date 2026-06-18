package com.malaria.android.ui.settings

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.data.OnboardingLanguage
import com.malaria.android.ui.locals.LocalSettingsStore

/**
 * Language picker section (spec §11). Editable; the edit screen triggers
 * a fresh biometric prompt — spec §11 calls this out explicitly to prevent
 * stranger-flips.
 */
@Composable
fun LanguageSection(navigator: SettingsNavigator) {
    val settings = LocalSettingsStore.current
    val language by settings.language.collectAsStateWithLifecycle()

    SectionScaffold(
        header = "Language",
        footer = "Changing language requires a fresh biometric prompt.",
    ) {
        EditableRow(
            label = "Language",
            value = OnboardingLanguage.fromCanonical(language).displayName,
            enabled = true,
            onClick = { navigator.push(SettingsDestination.EditLanguage) },
        )
    }
}
