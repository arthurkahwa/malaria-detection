import Foundation

/// Canonical audit-action vocabulary (spec §8). Stored on `AuditEntry.action`
/// as the lowercase-snake canonical string regardless of UI locale; UI
/// translates to localized display strings at render time.
///
/// Adding a value is a versioning event documented in `docs/SCHEMA.md`.
/// Removing one would be breaking and is not done — old logs continue to
/// contain the old values.
enum AuditAction: String, CaseIterable, Sendable {
    // Onboarding and lifecycle
    case adminProvisioningStarted = "admin_provisioning_started"
    case clinicConfigured = "clinic_configured"
    case inferencePolicySet = "inference_policy_set"
    case adminBiometricEnrolled = "admin_biometric_enrolled"
    case adminProvisioningCompleted = "admin_provisioning_completed"
    case microscopistClaimStarted = "microscopist_claim_started"
    case microscopistBiometricEnrolled = "microscopist_biometric_enrolled"
    case microscopistClaimCompleted = "microscopist_claim_completed"
    case deviceReprovisioned = "device_reprovisioned"
    case roleTransferred = "role_transferred"
    case profileUpdated = "profile_updated"

    // Authentication
    case authSuccess = "auth_success"
    case authFailure = "auth_failure"
    case sessionUnlocked = "session_unlocked"
    case sessionRelockedBackground = "session_relocked_background"
    case sessionRelockedManual = "session_relocked_manual"
    case sessionRelockedTimeout = "session_relocked_timeout"

    // Models
    case modelDownloadInitiated = "model_download_initiated"
    case modelDownloadCompleted = "model_download_completed"
    case modelDownloadFailed = "model_download_failed"
    case modelDownloadHashMismatch = "model_download_hash_mismatch"
    case modelCacheCleared = "model_cache_cleared"
    case activeModelChanged = "active_model_changed"

    // Inference
    case predictionCreated = "prediction_created"
    case predictionViewed = "prediction_viewed"
    case overrideRecorded = "override_recorded"
    case predictionLinkedAsDuplicate = "prediction_linked_as_duplicate"
    case sessionRelabeled = "session_relabeled"

    // Data management
    case exportInitiated = "export_initiated"
    case exportCompleted = "export_completed"
    case exportFailed = "export_failed"
    case crashLogShared = "crash_log_shared"

    // Configuration
    case thresholdChanged = "threshold_changed"
    case defaultModelChanged = "default_model_changed"
    case autoLogoutChanged = "auto_logout_changed"
    case languageChanged = "language_changed"

    /// Canonical English string used in persistence and exports.
    var canonical: String { rawValue }
}
