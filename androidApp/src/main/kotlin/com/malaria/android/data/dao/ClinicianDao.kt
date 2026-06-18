package com.malaria.android.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.malaria.android.data.entities.ClinicianProfile
import kotlinx.datetime.Clock
import java.util.UUID

/**
 * CRUD for [ClinicianProfile]. v1 is single-clinician (spec §9) so the DAO
 * exposes a [current] shortcut returning the sole row; multi-clinician
 * support is a v2 schema migration.
 *
 * Mirrors `iosApp/Persistence/ClinicianRepository.swift`.
 */
@Dao
abstract class ClinicianDao {

    /** The device's clinician, or null if onboarding hasn't completed. */
    @Query("SELECT * FROM clinician_profiles ORDER BY enrolled_at ASC LIMIT 1")
    abstract suspend fun current(): ClinicianProfile?

    /**
     * Enroll a brand-new clinician with a freshly generated `actorId` (which
     * also serves as the row primary key). `biometricEnrolled = false` until
     * [markBiometricEnrolled] runs.
     */
    suspend fun enroll(role: String, initials: String? = null): ClinicianProfile {
        val actorId = UUID.randomUUID().toString()
        val profile = ClinicianProfile(
            id = actorId,
            actorId = actorId,
            role = role,
            initials = initials,
            enrolledAt = Clock.System.now(),
            biometricEnrolled = false,
        )
        insert(profile)
        return profile
    }

    @Insert(onConflict = OnConflictStrategy.ABORT)
    protected abstract suspend fun insert(profile: ClinicianProfile)

    @Query("UPDATE clinician_profiles SET initials = :initials WHERE id = :id")
    abstract suspend fun updateInitials(id: String, initials: String?)

    @Query("UPDATE clinician_profiles SET biometric_enrolled = 1 WHERE id = :id")
    abstract suspend fun markBiometricEnrolled(id: String)

    /**
     * Wipe all clinician rows. Used by `Settings → Reset device` per
     * spec §10. Audit entries are preserved — only clinician-level
     * state is removed.
     */
    @Query("DELETE FROM clinician_profiles")
    abstract suspend fun wipe()
}
