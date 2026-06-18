package com.malaria.export

import kotlinx.serialization.Serializable

/**
 * Serializable mirror of the platform `ClinicianProfile` entity (spec §14).
 * Fields match the spec's schema block exactly. Timestamps are ISO-8601 UTC
 * strings.
 */
@Serializable
data class ExportedClinicianProfile(
    val actorId: String,
    val role: String,
    val initials: String?,
    val enrolledAt: String,
    val biometricEnrolled: Boolean,
)
