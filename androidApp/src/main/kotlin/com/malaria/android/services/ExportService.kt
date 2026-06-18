package com.malaria.android.services

import android.content.ContentResolver
import android.content.Context
import android.provider.Settings
import com.malaria.android.data.BuildEnvironment
import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.dao.ConsentDao
import com.malaria.android.data.dao.PredictionDao
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.data.entities.Prediction
import com.malaria.export.ExportBundleBuilder
import com.malaria.export.ExportPlatform
import com.malaria.export.ExportSummary
import com.malaria.export.ExportedAuditEntry
import com.malaria.export.ExportedClinicianProfile
import com.malaria.export.ExportedPrediction
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import java.io.File
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Phase 13 export-bundle service (spec §14). Mirrors
 * `iosApp/Services/ExportService.swift` step-for-step. The clinical-
 * correctness logic — JSON canonicalisation, HMAC signing, stable field
 * order — lives in the shared Kotlin module's
 * [com.malaria.export.ExportBundleBuilder]; this class is the Android-side
 * adapter that:
 *
 *   1. Loads typed inputs from the four DAOs + [SettingsStore].
 *   2. Converts Room entities to the shared `Exported*` DTOs (timestamps go
 *      through `Instant.toString()` so the format matches the iOS
 *      `ISO8601DateFormatter` form byte-for-byte).
 *   3. Calls `ExportBundleBuilder.build(...)` for the canonical signed JSON.
 *   4. Packs the JSON + a short `README.txt` into a ZIP via
 *      [java.util.zip.ZipOutputStream] and saves it under `context.cacheDir`.
 *   5. Emits the `export_initiated` / `export_completed` / `export_failed`
 *      audit chain per spec §14.
 *
 * The audit chain is the only side effect besides the temp-file write. The
 * caller (the DataManagementView composable) shares the resulting file via
 * `Intent.ACTION_SEND` + a `FileProvider`-issued content `Uri`.
 */
class ExportService(
    private val predictions: PredictionDao,
    private val audits: AuditDao,
    private val clinicians: ClinicianDao,
    private val consents: ConsentDao,
    private val settings: SettingsStore,
    private val auditLog: AuditLog,
    private val cacheRoot: File,
    private val clock: () -> Instant = { Clock.System.now() },
    private val deviceUuidProvider: () -> String,
) {

    /**
     * Production constructor wiring the device-scoped `Settings.Secure.ANDROID_ID`
     * UUID and the platform `Context.cacheDir`. Tests use the primary
     * constructor directly so neither dependency needs to be mocked.
     */
    constructor(
        context: Context,
        predictions: PredictionDao,
        audits: AuditDao,
        clinicians: ClinicianDao,
        consents: ConsentDao,
        settings: SettingsStore,
        auditLog: AuditLog,
    ) : this(
        predictions = predictions,
        audits = audits,
        clinicians = clinicians,
        consents = consents,
        settings = settings,
        auditLog = auditLog,
        cacheRoot = context.cacheDir,
        clock = { Clock.System.now() },
        deviceUuidProvider = { deriveDeviceUuid(context.contentResolver) },
    )

    private val _lastError = MutableStateFlow<Throwable?>(null)
    val lastError: StateFlow<Throwable?> = _lastError.asStateFlow()

    /** Produces a signed ZIP bundle and returns its [File]. */
    suspend fun generateBundle(): File {
        val actor = runCatching { clinicians.current() }.getOrNull()
        val actorId = actor?.actorId ?: "unknown"
        val actorRole = actor?.role ?: "unknown"

        auditLog.write(
            action = AuditAction.ExportInitiated,
            actorId = actorId,
            actorRoleAtTime = actorRole,
        )

        return try {
            val build = buildBundle(actorId = actorId)
            val zipFile = writeZip(build)
            val size = zipFile.length()
            auditLog.write(
                action = AuditAction.ExportCompleted,
                actorId = actorId,
                actorRoleAtTime = actorRole,
                metadata = mapOf(
                    "size" to size.toString(),
                    "signature" to build.signature,
                ),
            )
            _lastError.value = null
            zipFile
        } catch (t: Throwable) {
            _lastError.value = t
            auditLog.write(
                action = AuditAction.ExportFailed,
                actorId = actorId,
                actorRoleAtTime = actorRole,
                metadata = mapOf("reason" to (t.message ?: "unknown")),
            )
            throw t
        }
    }

    // -- Internals -------------------------------------------------------

    private data class BundleBuild(
        val signedJson: String,
        val readme: String,
        val signature: String,
        val filename: String,
    )

    private suspend fun buildBundle(actorId: String): BundleBuild {
        val clinicName = settings.clinicName.value
            ?: throw ExportException("Clinic configuration not found. Complete admin provisioning first.")
        val jurisdiction = settings.jurisdiction.value
            ?: throw ExportException("Jurisdiction not configured.")
        val lawfulBasis = settings.lawfulBasis.value
            ?: throw ExportException("Lawful basis not configured.")

        val now = clock()
        val exportTimestamp = now.toString()

        val predictionRows = predictions.recent(limit = Int.MAX_VALUE)
        val auditRows = audits.recent(limit = Int.MAX_VALUE)
        val clinicianRow = runCatching { clinicians.current() }.getOrNull()
        val consentRecords = if (clinicianRow != null) {
            runCatching { consents.records(clinicianRow.actorId) }.getOrDefault(emptyList())
        } else {
            emptyList()
        }

        val dtoPredictions = predictionRows.map { it.toDto() }
        val dtoAudits = auditRows.map { it.toDto() }
        val dtoProfiles = listOfNotNull(clinicianRow).map { it.toDto() }

        val summary = ExportSummary(
            predictionCount = predictionRows.size,
            sessionCount = predictionRows.map { it.sessionId }.toSet().size,
            auditEntryCount = auditRows.size,
            consentRecordCount = consentRecords.size,
            firstPredictionAt = predictionRows.minOfOrNull { it.timestamp }?.toString(),
            lastPredictionAt = predictionRows.maxOfOrNull { it.timestamp }?.toString(),
        )

        val deviceUuid = deviceUuidProvider()
        val signedJson = ExportBundleBuilder().build(
            exportTimestamp = exportTimestamp,
            exportedByActorId = actorId,
            deviceUuid = deviceUuid,
            platform = ExportPlatform.ANDROID,
            clinicName = clinicName,
            jurisdiction = jurisdiction,
            lawfulBasis = lawfulBasis,
            appVersion = BuildEnvironment.appVersion,
            osVersion = BuildEnvironment.osVersion,
            summary = summary,
            clinicianProfiles = dtoProfiles,
            predictions = dtoPredictions,
            auditLog = dtoAudits,
        )

        val signature = extractSignature(signedJson) ?: "unknown"
        val prefix = deviceUuid.take(8)
        val timestampSlug = exportTimestamp.replace(':', '-')
        val filename = "malaria-detector-export-$prefix-$timestampSlug.zip"

        val readme = buildReadme(
            exportTimestamp = exportTimestamp,
            clinicName = clinicName,
            summary = summary,
        )

        return BundleBuild(
            signedJson = signedJson,
            readme = readme,
            signature = signature,
            filename = filename,
        )
    }

    private fun writeZip(build: BundleBuild): File {
        val outputDir = File(cacheRoot, EXPORT_CACHE_SUBDIR).apply { mkdirs() }
        val zipFile = File(outputDir, build.filename)
        // Overwrite any stale file at the same name.
        if (zipFile.exists()) zipFile.delete()

        ZipOutputStream(zipFile.outputStream().buffered()).use { zip ->
            zip.putNextEntry(ZipEntry("export.json"))
            zip.write(build.signedJson.toByteArray(Charsets.UTF_8))
            zip.closeEntry()
            zip.putNextEntry(ZipEntry("README.txt"))
            zip.write(build.readme.toByteArray(Charsets.UTF_8))
            zip.closeEntry()
        }
        return zipFile
    }

    private fun extractSignature(json: String): String? {
        val marker = "\"signature\":\""
        val start = json.lastIndexOf(marker)
        if (start < 0) return null
        val from = start + marker.length
        val end = json.indexOf('"', from)
        if (end < 0) return null
        return json.substring(from, end)
    }

    private fun buildReadme(
        exportTimestamp: String,
        clinicName: String,
        summary: ExportSummary,
    ): String = """
        Malaria Detector export bundle
        ==============================
        Exported at:  $exportTimestamp
        Clinic:       $clinicName
        Predictions:  ${summary.predictionCount}
        Sessions:     ${summary.sessionCount}
        Audit rows:   ${summary.auditEntryCount}
        Consents:     ${summary.consentRecordCount}

        Contents:
          - export.json   — signed bundle (HMAC-SHA256, see spec §14)
          - README.txt    — this file

        Verifying the signature:
          1. Open export.json, strip the trailing `,"signature":"<hex>"}`.
          2. Re-serialise via the shared ExportBundleBuilder JSON config.
          3. Recompute the HMAC over the unsigned form using the device
             UUID + exportTimestamp salt.
          4. Compare to the original signature field.

        Generated by Malaria Detector v1. No images are included in this
        bundle — imageHash is the only durable trace of the analysed
        cells (spec §8).
    """.trimIndent()

    companion object {
        /** Subdirectory under `context.cacheDir` where bundles are stored. */
        const val EXPORT_CACHE_SUBDIR: String = "exports"

        /**
         * Derive the device UUID for the export bundle.
         *
         * `Settings.Secure.ANDROID_ID` is a 64-bit value scoped to the
         * signing key + user + device combination. It resets on factory
         * reset, which matches the spec §14 "device-reset semantics"
         * expectation. If the value is missing (rare; some emulators
         * return null), a fresh random UUID is generated — the bundle is
         * still verifiable; the only thing lost is per-device stability.
         */
        fun deriveDeviceUuid(resolver: ContentResolver): String {
            val androidId = runCatching {
                @Suppress("HardwareIds")
                Settings.Secure.getString(resolver, Settings.Secure.ANDROID_ID)
            }.getOrNull()
            return if (!androidId.isNullOrBlank()) androidId else UUID.randomUUID().toString()
        }
    }
}

/** Thrown when the bundle cannot be assembled (e.g. clinic not configured). */
class ExportException(message: String) : RuntimeException(message)

// -- DTO conversion -------------------------------------------------------

private fun Prediction.toDto(): ExportedPrediction = ExportedPrediction(
    id = id,
    sessionId = sessionId,
    timestamp = timestamp.toString(),
    modelId = modelId,
    modelVersion = modelVersion,
    parasitizedProb = parasitizedProb,
    uninfectedProb = uninfectedProb,
    label = label,
    threshold = threshold,
    flaggedForReview = flaggedForReview,
    inferenceMs = inferenceMs,
    imageHash = imageHash,
    clinicianOverride = clinicianOverride,
    overrideContext = overrideContext,
    duplicateOfId = duplicateOfId,
    sessionLabel = sessionLabel,
    appVersion = appVersion,
    osVersion = osVersion,
)

private fun AuditEntry.toDto(): ExportedAuditEntry = ExportedAuditEntry(
    id = id,
    seq = seq,
    timestamp = timestamp.toString(),
    actorId = actorId,
    actorRoleAtTime = actorRoleAtTime,
    action = action,
    resourceType = resourceType,
    resourceId = resourceId,
    metadataJson = metadataJson,
    overrideContext = overrideContext,
    overrideReason = overrideReason,
    overrideNotes = overrideNotes,
    contextReviewed = contextReviewed,
    overrideActorInitials = overrideActorInitials,
    appVersion = appVersion,
    osVersion = osVersion,
)

private fun ClinicianProfile.toDto(): ExportedClinicianProfile = ExportedClinicianProfile(
    actorId = actorId,
    role = role,
    initials = initials,
    enrolledAt = enrolledAt.toString(),
    biometricEnrolled = biometricEnrolled,
)
