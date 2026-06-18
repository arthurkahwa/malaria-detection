package com.malaria.export

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Builds the canonical export-bundle JSON (spec §14) — the sole shared-code
 * entry point used by both `iosApp/Services/ExportService.swift` and
 * `androidApp/.../services/ExportService.kt`.
 *
 * The same input produces the same JSON output on both platforms:
 *
 *  - field order is fixed by `@Serializable` declaration order;
 *  - `kotlinx.serialization` is configured with `prettyPrint = false`,
 *    `encodeDefaults = true`, and `explicitNulls = true` so the encoder
 *    emits `null` fields rather than dropping them (the spec schema lists
 *    nullable fields explicitly);
 *  - timestamps are pre-converted to ISO-8601 UTC strings by the platform
 *    caller, so no `Instant`/`Date` formatter divergence can sneak in;
 *  - the HMAC signature is computed over the canonical pre-signature JSON
 *    via [ExportSigner].
 *
 * Build twice with the same inputs → identical bytes. The
 * `ExportBundleBuilderTest.kt` asserts this property explicitly.
 */
class ExportBundleBuilder {

    /**
     * Produce the final signed bundle JSON.
     *
     * The platform caller assembles inputs (ISO-8601 timestamps, canonical
     * enum strings, the metadata-json blobs from the audit log) and this
     * method does the serialization + signing.
     */
    fun build(
        exportTimestamp: String,
        exportedByActorId: String,
        deviceUuid: String,
        platform: String,
        clinicName: String,
        jurisdiction: String,
        lawfulBasis: String,
        appVersion: String,
        osVersion: String,
        summary: ExportSummary,
        clinicianProfiles: List<ExportedClinicianProfile>,
        predictions: List<ExportedPrediction>,
        auditLog: List<ExportedAuditEntry>,
    ): String {
        val unsigned = UnsignedExportBundle(
            schemaVersion = EXPORT_SCHEMA_VERSION,
            exportTimestamp = exportTimestamp,
            exportedByActorId = exportedByActorId,
            deviceUuid = deviceUuid,
            platform = platform,
            clinicName = clinicName,
            jurisdiction = jurisdiction,
            lawfulBasis = lawfulBasis,
            appVersion = appVersion,
            osVersion = osVersion,
            summary = summary,
            clinicianProfiles = clinicianProfiles,
            predictions = predictions,
            auditLog = auditLog,
        )

        val unsignedJson = JSON.encodeToString(unsigned)
        val signature = ExportSigner.sign(
            unsignedJson = unsignedJson,
            deviceUuid = deviceUuid,
            timestampSalt = exportTimestamp,
        )

        val signed = ExportBundle(
            schemaVersion = EXPORT_SCHEMA_VERSION,
            exportTimestamp = exportTimestamp,
            exportedByActorId = exportedByActorId,
            deviceUuid = deviceUuid,
            platform = platform,
            clinicName = clinicName,
            jurisdiction = jurisdiction,
            lawfulBasis = lawfulBasis,
            appVersion = appVersion,
            osVersion = osVersion,
            summary = summary,
            clinicianProfiles = clinicianProfiles,
            predictions = predictions,
            auditLog = auditLog,
            signature = signature,
        )

        return JSON.encodeToString(signed)
    }

    companion object {
        /**
         * Canonical JSON configuration used by every export build. Exposed
         * for the test surface — verifiers can use the exact same encoder
         * to reproduce the unsigned form.
         */
        internal val JSON: Json = Json {
            prettyPrint = false
            encodeDefaults = true
            explicitNulls = true
        }
    }
}
