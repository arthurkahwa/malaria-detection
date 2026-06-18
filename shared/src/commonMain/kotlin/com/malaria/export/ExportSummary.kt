package com.malaria.export

import kotlinx.serialization.Serializable

/**
 * Aggregate counts + range markers for the export bundle's summary block
 * (spec §14). Timestamps are ISO-8601 UTC strings (or null) so the bundle is
 * portable across iOS / Android without re-serialising `Instant` / `Date`.
 *
 * The field order here is the declared order in the spec's example schema —
 * `kotlinx.serialization`'s default JSON encoder writes fields in declaration
 * order, which is the byte-identical-bundle invariant.
 */
@Serializable
data class ExportSummary(
    val predictionCount: Int,
    val sessionCount: Int,
    val auditEntryCount: Int,
    val consentRecordCount: Int,
    val firstPredictionAt: String?,
    val lastPredictionAt: String?,
)
