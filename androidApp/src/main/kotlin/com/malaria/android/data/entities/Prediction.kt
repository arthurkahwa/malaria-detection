package com.malaria.android.data.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import kotlinx.datetime.Instant
import java.util.UUID

/**
 * One row per inference event. Schema mirrors `docs/SCHEMA.md` and the iOS
 * SwiftData [`Prediction`](../../../../../../../../iosApp/Models/Prediction.swift)
 * model bit-for-bit.
 *
 * No image bytes ever land here — `imageHash` (SHA-256 of preprocessed
 * input) is the only durable trace of the analysed cell. Persisted under
 * SQLCipher AES-256 (spec §8). Column names are snake_case so the schema-
 * drift test has a stable string surface.
 */
@Entity(
    tableName = "predictions",
    indices = [
        Index("session_id"),
        Index("timestamp"),
        Index("flagged_for_review"),
    ],
)
data class Prediction(
    @PrimaryKey
    @ColumnInfo(name = "id")
    val id: String = UUID.randomUUID().toString(),

    @ColumnInfo(name = "session_id")
    val sessionId: String,

    @ColumnInfo(name = "timestamp")
    val timestamp: Instant,

    @ColumnInfo(name = "model_id")
    val modelId: String,

    @ColumnInfo(name = "model_version")
    val modelVersion: String,

    @ColumnInfo(name = "parasitized_prob")
    val parasitizedProb: Double,

    @ColumnInfo(name = "uninfected_prob")
    val uninfectedProb: Double,

    @ColumnInfo(name = "label")
    val label: String,

    @ColumnInfo(name = "threshold")
    val threshold: Double,

    @ColumnInfo(name = "flagged_for_review")
    val flaggedForReview: Boolean,

    @ColumnInfo(name = "inference_ms")
    val inferenceMs: Int,

    @ColumnInfo(name = "image_hash")
    val imageHash: String,

    @ColumnInfo(name = "clinician_override")
    val clinicianOverride: String? = null,

    @ColumnInfo(name = "override_context")
    val overrideContext: String? = null,

    @ColumnInfo(name = "duplicate_of_id")
    val duplicateOfId: String? = null,

    @ColumnInfo(name = "session_label")
    val sessionLabel: String? = null,

    @ColumnInfo(name = "app_version")
    val appVersion: String,

    @ColumnInfo(name = "os_version")
    val osVersion: String,
)
