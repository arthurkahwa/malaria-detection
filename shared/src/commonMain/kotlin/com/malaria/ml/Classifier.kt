package com.malaria.ml

import com.malaria.domain.ImageInput
import com.malaria.domain.Prediction
import kotlin.coroutines.cancellation.CancellationException

/**
 * Platform-provided image classifier.
 *
 * Each platform supplies an `actual class Classifier`:
 * - iOS (`iosMain`): wraps Core ML and Vision
 * - Android (`androidMain`): wraps LiteRT / TensorFlow Lite with GPU/NPU delegate
 *
 * The interface is intentionally edge/cloud-agnostic so that a future cloud
 * variant (`CloudClassifier`, deferred) extends this contract rather than
 * modifying it.
 */
expect class Classifier(modelId: String) {
    /**
     * Classify [image] and return the raw [Prediction] DTO. Throws
     * [InferenceError] subtypes on failure.
     *
     * Returning a thrown error rather than wrapping in `kotlin.Result`
     * is a Swift-bridge concession: Kotlin's `Result` is an inline class
     * that surfaces as an opaque box on the Obj-C side, with no
     * exposed `getOrThrow` / `isSuccess` accessors. Throwing maps
     * directly to Swift's `async throws`.
     */
    @Throws(InferenceError::class, CancellationException::class)
    suspend fun classify(image: ImageInput): Prediction

    /** Releases native resources (model handle, delegate, etc.). */
    fun close()
}
