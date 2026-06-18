package com.malaria.android.data.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import kotlinx.datetime.Instant
import java.util.UUID

/**
 * Onboarding acknowledgements (Hippocratic License, medical-device
 * disclaimer, lawful-basis selection, jurisdiction selection). Each record
 * pairs with an [AuditEntry] recording the same event — the audit log is the
 * chain-of-custody, the consent record is the per-clinician acknowledgement
 * that supports a later "did this user accept document version X" query.
 *
 * `documentVersion` locks the acknowledgement to specific shown wording so
 * a later legal-text revision doesn't retroactively claim prior consent.
 *
 * Mirrors `iosApp/Models/ConsentRecord.swift`.
 */
@Entity(
    tableName = "consent_records",
    indices = [
        Index("actor_id"),
        Index("consent_type"),
    ],
)
data class ConsentRecord(
    @PrimaryKey
    @ColumnInfo(name = "id")
    val id: String = UUID.randomUUID().toString(),

    @ColumnInfo(name = "timestamp")
    val timestamp: Instant,

    @ColumnInfo(name = "actor_id")
    val actorId: String,

    @ColumnInfo(name = "consent_type")
    val consentType: String,

    @ColumnInfo(name = "document_version")
    val documentVersion: String,

    @ColumnInfo(name = "value")
    val value: String,

    @ColumnInfo(name = "app_version")
    val appVersion: String,
)

/**
 * Canonical consent-type vocabulary mirroring iOS `ConsentType` in
 * `iosApp/Persistence/ConsentRepository.swift`.
 */
enum class ConsentType(val canonical: String) {
    HippocraticLicense("hippocratic_license"),
    MedicalDisclaimer("medical_disclaimer"),
    LawfulBasis("lawful_basis"),
    Jurisdiction("jurisdiction"),
}
