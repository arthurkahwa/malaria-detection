package com.malaria.android.ui.locals

import androidx.compose.runtime.compositionLocalOf
import com.malaria.android.data.MalariaDatabase
import com.malaria.android.services.AuditLog
import com.malaria.android.services.AuthGate
import com.malaria.android.services.CameraService
import com.malaria.android.services.ClassifierService
import com.malaria.android.services.CrashLogStore
import com.malaria.android.services.ExportService
import com.malaria.android.services.ModelDownloadService
import com.malaria.android.services.ModelRegistryService
import com.malaria.android.services.OnboardingState
import com.malaria.android.services.PredictionStore
import com.malaria.android.services.ResetDeviceCoordinator
import com.malaria.android.services.SettingsStore

// Per spec §19. Each service has its own CompositionLocal. The default lambda
// throws to make missing wiring fail loudly in development.

val LocalClassifier = compositionLocalOf<ClassifierService> {
    error("LocalClassifier not provided. Wrap in CompositionLocalProvider at root.")
}

val LocalAuthGate = compositionLocalOf<AuthGate> {
    error("LocalAuthGate not provided.")
}

val LocalCameraService = compositionLocalOf<CameraService> {
    error("LocalCameraService not provided.")
}

val LocalModelRegistry = compositionLocalOf<ModelRegistryService> {
    error("LocalModelRegistry not provided.")
}

val LocalPredictionStore = compositionLocalOf<PredictionStore> {
    error("LocalPredictionStore not provided.")
}

val LocalAuditLog = compositionLocalOf<AuditLog> {
    error("LocalAuditLog not provided.")
}

val LocalOnboardingState = compositionLocalOf<OnboardingState> {
    error("LocalOnboardingState not provided.")
}

/**
 * Direct database handle. Phase 10 (history viewer) needs reactive Flow
 * access to DAOs that the high-level services (`PredictionStore`,
 * `AuditLog`) intentionally don't expose. Exposing the database itself
 * here is the smallest change: composables can pick whichever DAO they
 * need and the existing service surfaces stay untouched.
 *
 * Mirrors the iOS pattern of `@Query` reading directly from the
 * `modelContext` for read-only history views; the underlying repos
 * (`PredictionRepository`, `AuditLogRepository`) are write-path on both
 * platforms.
 */
val LocalDatabase = compositionLocalOf<MalariaDatabase> {
    error("LocalDatabase not provided. Wrap in CompositionLocalProvider at root.")
}

/** Phase 11 — Settings facade backed by the audit log. */
val LocalSettingsStore = compositionLocalOf<SettingsStore> {
    error("LocalSettingsStore not provided.")
}

/** Phase 11 — Reset device coordinator (clinician wipe + onboarding reset). */
val LocalResetDeviceCoordinator = compositionLocalOf<ResetDeviceCoordinator> {
    error("LocalResetDeviceCoordinator not provided.")
}

/** Phase 13 — signed-ZIP export service (spec §14). */
val LocalExportService = compositionLocalOf<ExportService> {
    error("LocalExportService not provided.")
}

/** Phase 14 — on-device crash log store (spec §16). */
val LocalCrashLogStore = compositionLocalOf<CrashLogStore> {
    error("LocalCrashLogStore not provided.")
}

/** HF model downloader — download/progress/delete non-bundled models. */
val LocalModelDownloadService = compositionLocalOf<ModelDownloadService> {
    error("LocalModelDownloadService not provided.")
}
