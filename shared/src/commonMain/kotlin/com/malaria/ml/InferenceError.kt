package com.malaria.ml

/**
 * Errors that can be returned from [Classifier.classify] inside a failed
 * `Result`. Modeled as a sealed class so each platform can exhaustively
 * pattern-match.
 */
sealed class InferenceError(message: String, cause: Throwable? = null) : Exception(message, cause) {

    /** The model was not loaded (or has been closed) when [Classifier.classify] was invoked. */
    class ModelNotLoaded : InferenceError("Model is not loaded")

    /** The input image is unusable. [reason] explains why. */
    class InvalidImage(val reason: String) : InferenceError("Invalid image: $reason")

    /** Inference threw at the native layer. The native exception is in [cause]. */
    class InferenceFailed(cause: Throwable) : InferenceError("Inference failed: ${cause.message}", cause)

    /** The model binary for [modelId] is missing from bundle and cache. */
    class ModelFileMissing(val modelId: String) : InferenceError("Model file missing for id: $modelId")
}
