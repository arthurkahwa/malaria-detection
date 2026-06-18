package com.malaria.android.data.entities

/**
 * Canonical audit-action vocabulary (spec §8). Stored on
 * [AuditEntry.action] as the lowercase-snake canonical string regardless of
 * UI locale. UI translates to localized display strings at render time.
 *
 * Mirrors `iosApp/Models/AuditAction.swift` value-for-value — cross-platform
 * audit-log comparability is a spec §8 invariant.
 *
 * Adding a value is a versioning event documented in `docs/SCHEMA.md`.
 * Removing one would be breaking and is not done — old logs continue to
 * contain the old values.
 */
enum class AuditAction(val canonical: String) {
    // Onboarding and lifecycle
    AdminProvisioningStarted("admin_provisioning_started"),
    ClinicConfigured("clinic_configured"),
    InferencePolicySet("inference_policy_set"),
    AdminBiometricEnrolled("admin_biometric_enrolled"),
    AdminProvisioningCompleted("admin_provisioning_completed"),
    MicroscopistClaimStarted("microscopist_claim_started"),
    MicroscopistBiometricEnrolled("microscopist_biometric_enrolled"),
    MicroscopistClaimCompleted("microscopist_claim_completed"),
    DeviceReprovisioned("device_reprovisioned"),
    RoleTransferred("role_transferred"),
    ProfileUpdated("profile_updated"),

    // Authentication
    AuthSuccess("auth_success"),
    AuthFailure("auth_failure"),
    SessionUnlocked("session_unlocked"),
    SessionRelockedBackground("session_relocked_background"),
    SessionRelockedManual("session_relocked_manual"),
    SessionRelockedTimeout("session_relocked_timeout"),

    // Models
    ModelDownloadInitiated("model_download_initiated"),
    ModelDownloadCompleted("model_download_completed"),
    ModelDownloadFailed("model_download_failed"),
    ModelDownloadHashMismatch("model_download_hash_mismatch"),
    ModelCacheCleared("model_cache_cleared"),
    ActiveModelChanged("active_model_changed"),

    // Inference
    PredictionCreated("prediction_created"),
    PredictionViewed("prediction_viewed"),
    OverrideRecorded("override_recorded"),
    PredictionLinkedAsDuplicate("prediction_linked_as_duplicate"),
    SessionRelabeled("session_relabeled"),

    // Data management
    ExportInitiated("export_initiated"),
    ExportCompleted("export_completed"),
    ExportFailed("export_failed"),
    CrashLogShared("crash_log_shared"),

    // Configuration
    ThresholdChanged("threshold_changed"),
    DefaultModelChanged("default_model_changed"),
    AutoLogoutChanged("auto_logout_changed"),
    LanguageChanged("language_changed"),
}
