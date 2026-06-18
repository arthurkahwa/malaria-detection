package com.malaria.android.services

import com.malaria.domain.ImageInput
import com.malaria.domain.Prediction
import com.malaria.ml.Classifier
import kotlinx.coroutines.CoroutineScope

/**
 * Thin app-side wrapper around the shared-module [Classifier] expect/actual.
 *
 * The composable layer talks to this service rather than to `Classifier`
 * directly, so the activity owns the underlying interpreter's lifetime and
 * tests can substitute a fake `ClassifierService` via CompositionLocal.
 */
class ClassifierService(
    modelId: String,
    @Suppress("unused") private val scope: CoroutineScope,
) {
    private val classifier: Classifier = Classifier(modelId)

    suspend fun classify(image: ImageInput): Result<Prediction> =
        runCatching { classifier.classify(image) }

    fun close() {
        classifier.close()
    }
}
