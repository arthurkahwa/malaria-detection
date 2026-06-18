package com.malaria.ml

import com.malaria.domain.ImageInput
import com.malaria.domain.Prediction
import com.malaria.preprocessing.resizeRGB
import com.malaria.preprocessing.sha256Hex
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.ObjCObjectVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.set
import kotlinx.cinterop.value
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.datetime.Clock
import platform.CoreML.MLDictionaryFeatureProvider
import platform.CoreML.MLFeatureValue
import platform.CoreML.MLModel
import platform.CoreML.MLModelConfiguration
import platform.CoreVideo.CVPixelBufferCreate
import platform.CoreVideo.CVPixelBufferGetBaseAddress
import platform.CoreVideo.CVPixelBufferGetBytesPerRow
import platform.CoreVideo.CVPixelBufferLockBaseAddress
import platform.CoreVideo.CVPixelBufferRefVar
import platform.CoreVideo.CVPixelBufferUnlockBaseAddress
import platform.CoreVideo.kCVPixelFormatType_32BGRA
import platform.Foundation.NSBundle
import platform.Foundation.NSError
import platform.Foundation.NSFileManager
import platform.Foundation.NSNumber
import platform.Foundation.NSURL
import platform.Foundation.NSUserDomainMask
import platform.Foundation.NSApplicationSupportDirectory
import platform.posix.memset

/**
 * Core ML implementation of [Classifier] for iOS. Loads the bundled
 * `Malaria_<modelId>.mlmodelc` (compiled from `.mlpackage` by Xcode at
 * build time) and runs inference against the image input declared by the
 * model schema.
 *
 * The bundled `BNLeaky_Keras` model has:
 *   - Input  `input_image`        — Image (Color 128 × 128 RGB)
 *   - Output `classLabel`         — String predicted class
 *   - Output `classLabel_probs`   — Dictionary (String → Double) softmax
 *
 * Phase 2 deliverable per spec §22.
 */
@OptIn(ExperimentalForeignApi::class)
actual class Classifier actual constructor(private val modelId: String) {

    private val model: MLModel?
    private val loadFailure: Throwable?

    init {
        val (loaded, failure) = loadBundledModel(modelId)
        this.model = loaded
        this.loadFailure = failure
    }

    @Throws(InferenceError::class, CancellationException::class)
    actual suspend fun classify(image: ImageInput): Prediction {
        val mdl = model ?: throw (loadFailure as? InferenceError
            ?: InferenceError.ModelFileMissing(modelId))
        if (image.width <= 0 || image.height <= 0 || image.rgbBytes.size != image.width * image.height * 3) {
            throw InferenceError.InvalidImage("ImageInput is malformed")
        }

        val resized = resizeRGB(image, MODEL_INPUT_SIZE)
        val pixelBuffer = createBGRAPixelBuffer(resized, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            ?: throw InferenceError.InvalidImage("CVPixelBufferCreate failed")
        val featureValue = MLFeatureValue.featureValueWithPixelBuffer(pixelBuffer)

        val input = memScoped {
            val errorPtr = alloc<ObjCObjectVar<NSError?>>()
            MLDictionaryFeatureProvider(
                dictionary = mapOf<Any?, Any?>("input_image" to featureValue),
                error = errorPtr.ptr
            ) ?: throw InferenceError.InferenceFailed(
                RuntimeException(errorPtr.value?.localizedDescription ?: "MLDictionaryFeatureProvider init failed")
            )
        }

        val startInstant = Clock.System.now()
        val output = try {
            memScoped {
                val errorPtr = alloc<ObjCObjectVar<NSError?>>()
                mdl.predictionFromFeatures(input = input, error = errorPtr.ptr)
                    ?: throw InferenceError.InferenceFailed(
                        RuntimeException(errorPtr.value?.localizedDescription ?: "Core ML prediction failed")
                    )
            }
        } catch (e: InferenceError) {
            throw e
        } catch (e: Throwable) {
            throw InferenceError.InferenceFailed(e)
        }
        val inferenceMs = (Clock.System.now() - startInstant).inWholeMilliseconds

        val probsFeature = output.featureValueForName(PROBS_OUTPUT_NAME)
            ?: throw InferenceError.InferenceFailed(
                RuntimeException("Missing '$PROBS_OUTPUT_NAME' in Core ML output")
            )
        val rawDict = probsFeature.dictionaryValue
        val parasitized = lookupProbability(rawDict, LABEL_PARASITIZED)
        val uninfected = lookupProbability(rawDict, LABEL_UNINFECTED)

        return Prediction(
            parasitizedProb = parasitized,
            uninfectedProb = uninfected,
            modelId = modelId,
            timestamp = startInstant,
            inferenceMs = inferenceMs,
            imageHash = sha256Hex(image.rgbBytes)
        )
    }

    actual fun close() {
        // MLModel is a CoreFoundation-backed Obj-C object; ARC manages it.
    }

    companion object {
        private const val MODEL_INPUT_SIZE = 128
        private const val MODEL_FILE_PREFIX = "Malaria_"
        private const val PROBS_OUTPUT_NAME = "classLabel_probs"
        private const val LABEL_PARASITIZED = "Parasitized"
        private const val LABEL_UNINFECTED = "Uninfected"

        /**
         * Locate `Malaria_<modelId>.mlmodelc`. Checks the application support
         * directory first (downloaded models), then falls back to the main
         * bundle (compiled bundled models). Returns the loaded model or a
         * [Throwable] describing why load failed.
         */
        @OptIn(ExperimentalForeignApi::class)
        private fun loadBundledModel(modelId: String): Pair<MLModel?, Throwable?> {
            val filename = "$MODEL_FILE_PREFIX$modelId"

            // Check application support directory first (downloaded models).
            val appSupportURL = NSFileManager.defaultManager.URLsForDirectory(
                NSApplicationSupportDirectory, inDomains = NSUserDomainMask
            ).firstOrNull() as? NSURL
            val downloadedURL = appSupportURL
                ?.URLByAppendingPathComponent("MalariaDetector")
                ?.URLByAppendingPathComponent("models")
                ?.URLByAppendingPathComponent("$filename.mlmodelc")
            val url: NSURL? = if (downloadedURL != null &&
                NSFileManager.defaultManager.fileExistsAtPath(downloadedURL.path ?: "")) {
                downloadedURL
            } else {
                NSBundle.mainBundle.URLForResource(
                    name = filename,
                    withExtension = "mlmodelc"
                ) ?: NSBundle.mainBundle.URLForResource(
                    name = filename,
                    withExtension = "mlpackage"
                )
            }

            if (url == null) {
                return null to InferenceError.ModelFileMissing(modelId)
            }
            return memScoped {
                val errorPtr = alloc<ObjCObjectVar<NSError?>>()
                val config = MLModelConfiguration()
                val mlModel = MLModel.modelWithContentsOfURL(
                    url = url,
                    configuration = config,
                    error = errorPtr.ptr
                )
                if (mlModel != null) {
                    mlModel to null
                } else {
                    val err = errorPtr.value
                    null to InferenceError.InferenceFailed(
                        RuntimeException("Failed to load $filename: ${err?.localizedDescription ?: "unknown"}")
                    )
                }
            }
        }

        private fun lookupProbability(rawDict: Map<Any?, *>?, label: String): Float {
            if (rawDict == null) return 0f
            // Keys may surface as kotlin.String or NSString-bridged; values as NSNumber.
            for ((k, v) in rawDict) {
                val keyMatches = (k as? String) == label
                if (keyMatches) {
                    return when (v) {
                        is NSNumber -> v.doubleValue.toFloat()
                        is Number -> v.toFloat()
                        else -> 0f
                    }
                }
            }
            return 0f
        }
    }
}

/**
 * Allocate a 32-bit BGRA `CVPixelBuffer` of the target size and copy the
 * supplied RGB bytes into it. Core ML's image input expects BGRA when
 * the model declares an Image input without an explicit pixel-format
 * constraint, which is the default for `.mlpackage` files produced by
 * the notebook's `coremltools` export step.
 *
 * `rgb` must be exactly `width * height * 3` bytes, channel order RGB.
 */
@OptIn(ExperimentalForeignApi::class)
private fun createBGRAPixelBuffer(rgb: ByteArray, width: Int, height: Int): platform.CoreVideo.CVPixelBufferRef? {
    require(rgb.size == width * height * 3) {
        "rgb byte buffer size ${rgb.size} does not match $width × $height × 3"
    }
    return memScoped {
        val pbRef = alloc<CVPixelBufferRefVar>()
        val status = CVPixelBufferCreate(
            allocator = null,
            width = width.toULong(),
            height = height.toULong(),
            pixelFormatType = kCVPixelFormatType_32BGRA,
            pixelBufferAttributes = null,
            pixelBufferOut = pbRef.ptr
        )
        val buffer = pbRef.value
        if (status != 0 || buffer == null) {
            return@memScoped null
        }

        CVPixelBufferLockBaseAddress(buffer, 0u)
        try {
            val base = CVPixelBufferGetBaseAddress(buffer)?.reinterpret<ByteVar>()
                ?: return@memScoped null
            val bytesPerRow = CVPixelBufferGetBytesPerRow(buffer).toInt()
            memset(base, 0, (bytesPerRow * height).toULong())

            for (y in 0 until height) {
                val rowStart = y * bytesPerRow
                val rgbRow = y * width * 3
                for (x in 0 until width) {
                    val src = rgbRow + x * 3
                    val dst = rowStart + x * 4
                    val r = rgb[src].toInt() and 0xFF
                    val g = rgb[src + 1].toInt() and 0xFF
                    val b = rgb[src + 2].toInt() and 0xFF
                    base[dst] = b.toByte()
                    base[dst + 1] = g.toByte()
                    base[dst + 2] = r.toByte()
                    base[dst + 3] = 0xFF.toByte()
                }
            }
        } finally {
            CVPixelBufferUnlockBaseAddress(buffer, 0u)
        }
        buffer
    }
}
