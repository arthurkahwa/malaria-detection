// Requires API 36 Android emulator with the TFLite native library available.
// Run via: ./gradlew :androidApp:connectedDebugAndroidTest
// NOT part of the JVM unit test suite — TFLite requires the native
// `libtensorflowlite_jni.so` which isn't available on the JVM, only on
// real devices / emulators.
package com.malaria.android.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.malaria.domain.ImageInput
import com.malaria.ml.Classifier
import com.malaria.ml.TFLiteContext
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end smoke test for the Android TFLite pipeline (spec §22 Phase 3).
 *
 * Builds a synthetic 128×128 RGB gray image, runs it through the bundled
 * `Malaria_BNLeaky_Keras` model, and asserts the output schema matches what
 * `PredictionStore` expects to map. Mirrors `iosApp/Tests/
 * CoreMLClassifierTests.swift` field-for-field so the cross-platform
 * contract is exercised on both sides.
 *
 * Probability values are not asserted because synthetic gray noise is not
 * a meaningful blood-cell image; we only verify model load, inference run,
 * and well-formed Parasitized/Uninfected probabilities that sum to ~1.0.
 */
@RunWith(AndroidJUnit4::class)
class TFLiteClassifierTest {

    @Before
    fun installContext() {
        // In production this happens in MalariaApplication.onCreate(); in
        // instrumented tests we install explicitly because the test runner
        // doesn't always go through the Application subclass.
        TFLiteContext.install(InstrumentationRegistry.getInstrumentation().targetContext)
    }

    @Test
    fun bnLeakyKerasClassifier_runsSyntheticImage_endToEnd() = runTest {
        val classifier = Classifier(modelId = "BNLeaky_Keras")
        try {
            // 128×128 RGB filled with mid-gray (0x80).
            val totalBytes = 128 * 128 * 3
            val bytes = ByteArray(totalBytes) { 0x80.toByte() }
            val image = ImageInput(rgbBytes = bytes, width = 128, height = 128)

            val prediction = classifier.classify(image)

            assertNotNull(prediction)
            assertEquals("BNLeaky_Keras", prediction.modelId)
            assertFalse("imageHash must not be empty", prediction.imageHash.isEmpty())
            assertTrue("inferenceMs must be non-negative", prediction.inferenceMs >= 0)

            val pSum = prediction.parasitizedProb + prediction.uninfectedProb
            assertEquals(
                "Softmax outputs must sum to 1 (got $pSum)",
                1.0,
                pSum.toDouble(),
                0.01
            )
            assertTrue(prediction.parasitizedProb >= 0f)
            assertTrue(prediction.parasitizedProb <= 1f)
            assertTrue(prediction.uninfectedProb >= 0f)
            assertTrue(prediction.uninfectedProb <= 1f)
        } finally {
            classifier.close()
        }
    }
}
