package com.malaria.android.data

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

/**
 * Persisted UI language for onboarding (spec §10 Phase 1 step 1).
 *
 * Stored in DataStore Preferences — unencrypted because language choice is
 * not sensitive and must survive a `Reset device` operation (spec §10
 * re-onboarding). Mirrors `iosApp/Persistence/LanguagePreference.swift`
 * value-for-value: canonical lowercase rawValues, English-only display
 * strings (per spec §15 the picker labels stay in English so a fresh-device
 * admin doesn't have to read e.g. "Swahili" rendered in Swahili).
 *
 * Phase 12 will add Swahili/French/Portuguese localised string resources;
 * for v0.1 only English is shipped, but the enum is already in place so the
 * picker step renders the full four-option list per spec.
 */
enum class OnboardingLanguage(val canonical: String, val displayName: String) {
    English("en", "English"),
    Swahili("sw", "Swahili"),
    French("fr", "French"),
    Portuguese("pt", "Portuguese");

    companion object {
        fun fromCanonical(value: String?): OnboardingLanguage =
            values().firstOrNull { it.canonical == value } ?: English
    }
}

private val Context.languageDataStore by preferencesDataStore(name = "onboarding_language")
private val LANGUAGE_KEY = stringPreferencesKey("onboarding.language")

/**
 * DataStore-backed wrapper around the language preference. Stable across
 * resets per spec §10.
 *
 * Compose call sites collect [flow] with `collectAsStateWithLifecycle` and
 * write changes via [set].
 */
class LanguagePreference(private val context: Context) {

    val flow: Flow<OnboardingLanguage> = context.languageDataStore.data.map { prefs: Preferences ->
        OnboardingLanguage.fromCanonical(prefs[LANGUAGE_KEY])
    }

    suspend fun get(): OnboardingLanguage = flow.first()

    suspend fun set(language: OnboardingLanguage) {
        context.languageDataStore.edit { prefs ->
            prefs[LANGUAGE_KEY] = language.canonical
        }
    }
}
