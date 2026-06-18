package com.malaria.export

import kotlinx.serialization.Serializable

/**
 * Top-level export bundle (spec §14). Field order is the spec's declared
 * order; the JSON encoder writes fields in declaration order, which keeps
 * the serialized form byte-identical between iOS and Android exports of the
 * same content.
 *
 * Two-stage construction: [ExportBundleBuilder] serializes everything
 * *except* `signature` first, computes the HMAC over that canonical-form
 * JSON via [ExportSigner], then emits the final bundle JSON with the
 * signature glued on. Verifiers do the inverse — strip the signature,
 * re-serialize, recompute, compare.
 */
@Serializable
data class ExportBundle(
    val schemaVersion: String,
    val exportTimestamp: String,
    val exportedByActorId: String,
    val deviceUuid: String,
    val platform: String,
    val clinicName: String,
    val jurisdiction: String,
    val lawfulBasis: String,
    val appVersion: String,
    val osVersion: String,
    val summary: ExportSummary,
    val clinicianProfiles: List<ExportedClinicianProfile>,
    val predictions: List<ExportedPrediction>,
    val auditLog: List<ExportedAuditEntry>,
    val signature: String,
)

/**
 * Same shape as [ExportBundle] minus the trailing `signature` field. Serializing
 * this is the canonical pre-signature form over which [ExportSigner.sign]
 * computes its HMAC.
 *
 * Field-for-field parity with [ExportBundle] (apart from the missing
 * `signature`) is checked by `ExportBundleBuilderTest.kt`.
 */
@Serializable
internal data class UnsignedExportBundle(
    val schemaVersion: String,
    val exportTimestamp: String,
    val exportedByActorId: String,
    val deviceUuid: String,
    val platform: String,
    val clinicName: String,
    val jurisdiction: String,
    val lawfulBasis: String,
    val appVersion: String,
    val osVersion: String,
    val summary: ExportSummary,
    val clinicianProfiles: List<ExportedClinicianProfile>,
    val predictions: List<ExportedPrediction>,
    val auditLog: List<ExportedAuditEntry>,
)

/**
 * Canonical schema version emitted in every bundle. Bump on any schema
 * change.
 */
const val EXPORT_SCHEMA_VERSION: String = "1.0"

/**
 * Canonical platform tags emitted on the `platform` field. The two strings
 * are the only legal values per spec §14.
 */
object ExportPlatform {
    const val IOS: String = "ios"
    const val ANDROID: String = "android"
}
