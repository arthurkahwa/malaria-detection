package com.malaria.domain

import kotlinx.datetime.Instant

/**
 * Raw inference result and metadata produced by a [com.malaria.ml.Classifier].
 *
 * This DTO is *not* a persisted entity. Each platform maps it to its own
 * persistence type (SwiftData `@Model` on iOS, Room `@Entity` on Android) and
 * assigns an `id`, `sessionId`, and `flaggedForReview` there.
 */
data class Prediction(
    val parasitizedProb: Float,
    val uninfectedProb: Float,
    val modelId: String,
    val timestamp: Instant,
    val inferenceMs: Long,
    val imageHash: String
)
