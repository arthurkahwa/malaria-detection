package com.malaria.ml

import android.content.Context

/**
 * Holder for the Android Application Context that the [Classifier] uses
 * to read the bundled `.tflite` file out of `assets/`.
 *
 * The expect/actual signature `Classifier(modelId: String)` is platform-
 * agnostic — it cannot take a [Context] argument without breaking iOS
 * parity. We work around that by stashing the application context here
 * at app startup and reading it back when the classifier is constructed.
 *
 * Lifecycle: [install] must be called before constructing any
 * [Classifier]; otherwise the constructor throws
 * [InferenceError.ModelFileMissing]. The canonical install site is
 * `MalariaApplication.onCreate()`.
 *
 * Spec §22 Phase 3 deliverable.
 */
object TFLiteContext {

    @Volatile
    var applicationContext: Context? = null
        internal set

    /**
     * Stash the application-scoped context for later read by
     * [Classifier]. Idempotent and safe to call more than once.
     */
    fun install(context: Context) {
        applicationContext = context.applicationContext
    }
}
