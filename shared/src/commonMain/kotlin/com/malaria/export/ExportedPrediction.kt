package com.malaria.export

import kotlinx.serialization.Serializable

/**
 * Serializable mirror of the platform `Prediction` entity (spec §14). Every
 * field from the persistence row appears here in the spec's declaration
 * order; `timestamp` is the ISO-8601 UTC string form of the persisted
 * `Instant` / `Date`.
 *
 * Fields match `iosApp/Models/Prediction.swift` and
 * `androidApp/.../entities/Prediction.kt` value-for-value so an exporting
 * platform's bundle is byte-identical to the other platform's bundle of the
 * same content.
 */
@Serializable
data class ExportedPrediction(
    val id: String,
    val sessionId: String,
    val timestamp: String,
    val modelId: String,
    val modelVersion: String,
    val parasitizedProb: Double,
    val uninfectedProb: Double,
    val label: String,
    val threshold: Double,
    val flaggedForReview: Boolean,
    val inferenceMs: Int,
    val imageHash: String,
    val clinicianOverride: String?,
    val overrideContext: String?,
    val duplicateOfId: String?,
    val sessionLabel: String?,
    val appVersion: String,
    val osVersion: String,
)
