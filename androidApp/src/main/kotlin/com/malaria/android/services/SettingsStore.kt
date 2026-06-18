package com.malaria.android.services

import com.malaria.android.data.LanguagePreference
import com.malaria.android.data.OnboardingLanguage
import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.entities.AuditAction
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Settings facade backed by the audit log (spec §11, §8). Mirrors the iOS
 * `SettingsStore` value-for-value.
 *
 * Clinic-level config (name, jurisdiction, lawful basis) and the inference
 * policy (threshold, default model, auto-logout) live in the audit log as
 * the source of truth — written during Phase 1 admin provisioning
 * (`clinic_configured`, `inference_policy_set`) and updated by Settings
 * edits via `threshold_changed`, `default_model_changed`,
 * `auto_logout_changed`. Holding the latest values here avoids re-parsing
 * metadata JSON on every render.
 *
 * `language` lives in [LanguagePreference] (DataStore) because it must
 * survive `Reset device` (spec §10) where the audit log gets a fresh
 * `device_reprovisioned` entry but keeps every prior provisioning entry —
 * the chain-of-custody requirement.
 *
 * `language` is a [String] (canonical lowercase) rather than the
 * `OnboardingLanguage` enum so the value matches the iOS surface field-
 * for-field — the composables convert via `OnboardingLanguage.fromCanonical`.
 *
 * `languagePreference` may be null in unit tests that want to assert just
 * the audit-driven hydration without spinning up a DataStore context.
 */
class SettingsStore(
    private val auditDao: AuditDao,
    private val audit: AuditLog,
    private val clinicians: ClinicianDao,
    private val languagePreference: LanguagePreference? = null,
) {
    private val _clinicName = MutableStateFlow<String?>(null)
    val clinicName: StateFlow<String?> = _clinicName.asStateFlow()

    private val _jurisdiction = MutableStateFlow<String?>(null)
    val jurisdiction: StateFlow<String?> = _jurisdiction.asStateFlow()

    private val _lawfulBasis = MutableStateFlow<String?>(null)
    val lawfulBasis: StateFlow<String?> = _lawfulBasis.asStateFlow()

    private val _threshold = MutableStateFlow(0.3)
    val threshold: StateFlow<Double> = _threshold.asStateFlow()

    private val _defaultModelId = MutableStateFlow("BNLeaky_Keras")
    val defaultModelId: StateFlow<String> = _defaultModelId.asStateFlow()

    private val _autoLogoutMinutes = MutableStateFlow(15)
    val autoLogoutMinutes: StateFlow<Int> = _autoLogoutMinutes.asStateFlow()

    private val _language = MutableStateFlow(OnboardingLanguage.English.canonical)
    val language: StateFlow<String> = _language.asStateFlow()

    /**
     * Hydrate from the audit log + [LanguagePreference]. Called once at
     * composition root; safe to call again after a `Reset device`.
     */
    suspend fun hydrate() {
        runCatching {
            auditDao.entries(AuditAction.ClinicConfigured.canonical).firstOrNull()
        }.getOrNull()?.let { entry ->
            val metadata = decode(entry.metadataJson)
            _clinicName.value = metadata["clinic_name"]
            _jurisdiction.value = metadata["jurisdiction"]
            _lawfulBasis.value = metadata["lawful_basis"]
        } ?: run {
            _clinicName.value = null
            _jurisdiction.value = null
            _lawfulBasis.value = null
        }

        runCatching {
            auditDao.entries(AuditAction.InferencePolicySet.canonical).firstOrNull()
        }.getOrNull()?.let { entry ->
            val metadata = decode(entry.metadataJson)
            metadata["threshold"]?.toDoubleOrNull()?.let { _threshold.value = it }
            metadata["default_model"]?.let { _defaultModelId.value = it }
            metadata["auto_logout_minutes"]?.toIntOrNull()?.let { _autoLogoutMinutes.value = it }
        }

        // Walk audit log for the most-recent change-of-value entries so a
        // post-onboarding edit wins over the original `inference_policy_set`.
        runCatching {
            auditDao.entries(AuditAction.ThresholdChanged.canonical).firstOrNull()
        }.getOrNull()?.let { entry ->
            decode(entry.metadataJson)["new_value"]?.toDoubleOrNull()?.let { _threshold.value = it }
        }
        runCatching {
            auditDao.entries(AuditAction.DefaultModelChanged.canonical).firstOrNull()
        }.getOrNull()?.let { entry ->
            decode(entry.metadataJson)["new_value"]?.let { _defaultModelId.value = it }
        }
        runCatching {
            auditDao.entries(AuditAction.AutoLogoutChanged.canonical).firstOrNull()
        }.getOrNull()?.let { entry ->
            decode(entry.metadataJson)["new_value"]?.toIntOrNull()?.let { _autoLogoutMinutes.value = it }
        }

        languagePreference?.let { _language.value = it.get().canonical }
    }

    // -- Edits ------------------------------------------------------------

    suspend fun updateThreshold(newValue: Double) {
        val old = _threshold.value
        _threshold.value = newValue
        val actor = currentActor()
        audit.write(
            action = AuditAction.ThresholdChanged,
            actorId = actor.id,
            actorRoleAtTime = actor.role,
            metadata = mapOf("old_value" to old.toString(), "new_value" to newValue.toString()),
        )
    }

    suspend fun updateDefaultModel(newValue: String) {
        val old = _defaultModelId.value
        _defaultModelId.value = newValue
        val actor = currentActor()
        audit.write(
            action = AuditAction.DefaultModelChanged,
            actorId = actor.id,
            actorRoleAtTime = actor.role,
            metadata = mapOf("old_value" to old, "new_value" to newValue),
        )
    }

    suspend fun updateAutoLogoutMinutes(newValue: Int) {
        val old = _autoLogoutMinutes.value
        _autoLogoutMinutes.value = newValue
        val actor = currentActor()
        audit.write(
            action = AuditAction.AutoLogoutChanged,
            actorId = actor.id,
            actorRoleAtTime = actor.role,
            metadata = mapOf("old_value" to old.toString(), "new_value" to newValue.toString()),
        )
    }

    suspend fun updateLanguage(newValue: OnboardingLanguage) {
        val old = _language.value
        _language.value = newValue.canonical
        languagePreference?.set(newValue)
        val actor = currentActor()
        audit.write(
            action = AuditAction.LanguageChanged,
            actorId = actor.id,
            actorRoleAtTime = actor.role,
            metadata = mapOf("old_value" to old, "new_value" to newValue.canonical),
        )
    }

    suspend fun updateInitials(newValue: String?) {
        val profile = runCatching { clinicians.current() }.getOrNull() ?: return
        clinicians.updateInitials(profile.id, newValue)
        audit.write(
            action = AuditAction.ProfileUpdated,
            actorId = profile.actorId,
            actorRoleAtTime = profile.role,
            resourceType = "clinician",
            resourceId = profile.actorId,
            metadata = mapOf("field" to "initials"),
        )
    }

    // -- Private ----------------------------------------------------------

    private data class Actor(val id: String, val role: String)

    private suspend fun currentActor(): Actor {
        val profile = runCatching { clinicians.current() }.getOrNull()
        return Actor(profile?.actorId ?: "unknown", profile?.role ?: "unknown")
    }

    /**
     * Parse the same flat string-map shape that `encodeMetadata` writes
     * in `AuditLog.kt`. Lenient — unknown/malformed JSON yields an empty
     * map so callers can fall back to defaults.
     */
    private fun decode(json: String): Map<String, String> {
        if (json.isBlank() || json == "{}") return emptyMap()
        // Tiny parser scoped to the shape AuditLog actually writes:
        // a flat string-to-string object with escaped string values.
        val out = mutableMapOf<String, String>()
        var i = 0
        val s = json
        if (s.isEmpty() || s[0] != '{') return emptyMap()
        i = 1
        while (i < s.length) {
            // Skip whitespace
            while (i < s.length && s[i].isWhitespace()) i++
            if (i < s.length && s[i] == '}') return out
            // Expect key
            if (i >= s.length || s[i] != '"') return out
            i++
            val keyBuilder = StringBuilder()
            while (i < s.length && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.length) {
                    keyBuilder.append(unescape(s[i + 1]))
                    i += 2
                } else {
                    keyBuilder.append(s[i]); i++
                }
            }
            i++ // closing quote
            while (i < s.length && s[i].isWhitespace()) i++
            if (i >= s.length || s[i] != ':') return out
            i++
            while (i < s.length && s[i].isWhitespace()) i++
            if (i >= s.length || s[i] != '"') return out
            i++
            val valueBuilder = StringBuilder()
            while (i < s.length && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.length) {
                    valueBuilder.append(unescape(s[i + 1]))
                    i += 2
                } else {
                    valueBuilder.append(s[i]); i++
                }
            }
            i++ // closing quote
            out[keyBuilder.toString()] = valueBuilder.toString()
            while (i < s.length && s[i].isWhitespace()) i++
            if (i < s.length && s[i] == ',') { i++; continue }
            if (i < s.length && s[i] == '}') return out
            return out
        }
        return out
    }

    private fun unescape(c: Char): Char = when (c) {
        'n' -> '\n'
        'r' -> '\r'
        't' -> '\t'
        'b' -> '\b'
        'f' -> ''
        '"' -> '"'
        '\\' -> '\\'
        else -> c
    }
}
