package com.malaria.android.data.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey
import kotlinx.datetime.Instant
import java.util.UUID

/**
 * The device's single clinician (v1 is single-clinician per spec §9).
 * `actorId` is pseudonymous — never linked to PII inside the app. A clinic
 * admin keeps the UUID → person mapping outside the app's scope.
 *
 * Mirrors `iosApp/Models/ClinicianProfile.swift`.
 */
@Entity(tableName = "clinician_profiles")
data class ClinicianProfile(
    @PrimaryKey
    @ColumnInfo(name = "id")
    val id: String = UUID.randomUUID().toString(),

    @ColumnInfo(name = "actor_id")
    val actorId: String,

    @ColumnInfo(name = "role")
    val role: String,

    @ColumnInfo(name = "initials")
    val initials: String? = null,

    @ColumnInfo(name = "enrolled_at")
    val enrolledAt: Instant,

    @ColumnInfo(name = "biometric_enrolled")
    val biometricEnrolled: Boolean = false,
)
