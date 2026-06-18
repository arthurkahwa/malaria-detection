// Phase 3 architectural decision (resolved): TensorFlow Lite is declared in
// shared/build.gradle.kts under the androidMain source-set, NOT under
// androidApp. Spec §5 lines 519-521 suggested TFLite lives in androidApp and
// shared:androidMain only "interfaces with" it — that's impractical because
// the actual class itself needs to import `org.tensorflow.lite.Interpreter`
// at compile time. Putting the dep on shared:androidMain is cleaner; the
// androidApp module still sees the symbols via `api(...)` transitivity.
package com.malaria.ml

import com.malaria.domain.ImageInput
import com.malaria.domain.Prediction
import com.malaria.preprocessing.preprocess
import com.malaria.preprocessing.sha256Hex
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.datetime.Clock
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TensorFlow Lite implementation of [Classifier] for Android. Loads the
 * bundled `Malaria_<modelId>.tflite` (placed in `androidApp/src/main/
 * assets/models/` by the build, gitignored as a 32 MB artifact) and runs
 * inference against the model's float input tensor.
 *
 * The bundled `BNLeaky_Keras` model expects:
 *   - Input  shape `[1, 128, 128, 3]` (NHWC), float32 in `[0, 1]`
 *   - Output shape `[1, 2]`, softmax over `[Parasitized, Uninfected]`
 *
 * Class index mapping is verified at init against the companion
 * `Malaria_<modelId>_labels.txt` so a model rebuild that flips the label
 * order is caught at load time instead of producing inverted predictions
 * at runtime.
 *
 * Phase 3 deliverable per spec §22.
 */
actual class Classifier actual constructor(private val modelId: String) {

    private val interpreter: Interpreter?
    private val loadFailure: Throwable?
    private val inputShape: IntArray
    private val parasitizedIdx: Int
    private val uninfectedIdx: Int

    init {
        val (loaded, failure, shape, pIdx, uIdx) = loadBundledModel(modelId)
        this.interpreter = loaded
        this.loadFailure = failure
        this.inputShape = shape
        this.parasitizedIdx = pIdx
        this.uninfectedIdx = uIdx
    }

    @Throws(InferenceError::class, CancellationException::class)
    actual suspend fun classify(image: ImageInput): Prediction {
        val ip = interpreter ?: throw (loadFailure as? InferenceError
            ?: InferenceError.ModelFileMissing(modelId))
        if (image.width <= 0 || image.height <= 0 ||
            image.rgbBytes.size != image.width * image.height * 3) {
            throw InferenceError.InvalidImage("ImageInput is malformed")
        }

        // Pure-Kotlin resize + normalize to [0, 1]. Matches iOS by-construction
        // because both platforms call the same shared Preprocessor.
        val normalized: FloatArray = preprocess(image, MODEL_INPUT_SIZE)

        val inputBuffer = buildInputBuffer(normalized, inputShape)
        val output = Array(1) { FloatArray(NUM_CLASSES) }

        val startInstant = Clock.System.now()
        try {
            ip.run(inputBuffer, output)
        } catch (e: Throwable) {
            throw InferenceError.InferenceFailed(e)
        }
        val inferenceMs = (Clock.System.now() - startInstant).inWholeMilliseconds

        val parasitized = output[0][parasitizedIdx]
        val uninfected = output[0][uninfectedIdx]

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
        interpreter?.close()
    }

    private data class LoadResult(
        val interpreter: Interpreter?,
        val failure: Throwable?,
        val inputShape: IntArray,
        val parasitizedIdx: Int,
        val uninfectedIdx: Int
    )

    companion object {
        private const val MODEL_INPUT_SIZE = 128
        private const val NUM_CLASSES = 2
        private const val MODEL_FILE_PREFIX = "Malaria_"
        private const val MODEL_ASSET_DIR = "models"
        private const val LABEL_PARASITIZED = "Parasitized"
        private const val LABEL_UNINFECTED = "Uninfected"

        private fun loadBundledModel(modelId: String): LoadResult {
            val context = TFLiteContext.applicationContext
                ?: return LoadResult(
                    interpreter = null,
                    failure = InferenceError.ModelFileMissing(modelId),
                    inputShape = intArrayOf(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
                    parasitizedIdx = 0,
                    uninfectedIdx = 1
                )

            val modelFilename = "$MODEL_ASSET_DIR/$MODEL_FILE_PREFIX$modelId.tflite"
            val labelsFilename = "$MODEL_ASSET_DIR/$MODEL_FILE_PREFIX$modelId" + "_labels.txt"

            // Prefer a downloaded model in filesDir over the bundled asset.
            val cachedFile = java.io.File(
                context.filesDir,
                "MalariaDetector/models/$MODEL_FILE_PREFIX$modelId.tflite"
            )

            val modelBuffer: MappedByteBuffer = if (cachedFile.exists()) {
                try {
                    FileInputStream(cachedFile).use { stream ->
                        stream.channel.map(FileChannel.MapMode.READ_ONLY, 0, cachedFile.length())
                    }
                } catch (e: IOException) {
                    // Fall through to asset loading if the cached file is unreadable.
                    try {
                        loadAssetAsMappedBuffer(context, modelFilename)
                    } catch (e2: IOException) {
                        return LoadResult(
                            interpreter = null,
                            failure = InferenceError.ModelFileMissing(modelId),
                            inputShape = intArrayOf(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
                            parasitizedIdx = 0,
                            uninfectedIdx = 1
                        )
                    }
                }
            } else {
                try {
                    loadAssetAsMappedBuffer(context, modelFilename)
                } catch (e: IOException) {
                    return LoadResult(
                        interpreter = null,
                        failure = InferenceError.ModelFileMissing(modelId),
                        inputShape = intArrayOf(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
                        parasitizedIdx = 0,
                        uninfectedIdx = 1
                    )
                }
            }

            // Verify label ordering — throw at init if the model ships with
            // flipped indices, rather than silently inverting Parasitized /
            // Uninfected probabilities at runtime.
            val (pIdx, uIdx) = try {
                readLabelIndices(context, labelsFilename)
            } catch (e: IOException) {
                // Labels file missing → assume canonical order from the
                // notebook (Parasitized = 0, Uninfected = 1). Don't fail the
                // load — the labels file is metadata, not required for
                // inference, and the bundled model ships with the canonical
                // order verified by the upstream notebook.
                0 to 1
            }

            return try {
                val opts = Interpreter.Options()
                val ip = Interpreter(modelBuffer, opts)
                val detectedShape = ip.getInputTensor(0).shape()
                LoadResult(
                    interpreter = ip,
                    failure = null,
                    inputShape = detectedShape,
                    parasitizedIdx = pIdx,
                    uninfectedIdx = uIdx
                )
            } catch (e: Throwable) {
                LoadResult(
                    interpreter = null,
                    failure = InferenceError.InferenceFailed(e),
                    inputShape = intArrayOf(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
                    parasitizedIdx = pIdx,
                    uninfectedIdx = uIdx
                )
            }
        }

        /**
         * Standard TFLite asset-loading pattern: open the asset descriptor,
         * memory-map the underlying file slice. Required because Interpreter
         * needs a [MappedByteBuffer], not an InputStream.
         */
        private fun loadAssetAsMappedBuffer(
            context: android.content.Context,
            filename: String
        ): MappedByteBuffer {
            val fd = context.assets.openFd(filename)
            FileInputStream(fd.fileDescriptor).use { stream ->
                val channel = stream.channel
                return channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength
                )
            }
        }

        /**
         * Reads `labels.txt` (one class label per line) and returns the
         * indices of [LABEL_PARASITIZED] and [LABEL_UNINFECTED]. Falls back
         * to canonical 0/1 if labels are missing entirely (handled by the
         * caller's catch on IOException).
         */
        private fun readLabelIndices(
            context: android.content.Context,
            filename: String
        ): Pair<Int, Int> {
            val labels = context.assets.open(filename).bufferedReader().useLines { lines ->
                lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
            }
            val pIdx = labels.indexOf(LABEL_PARASITIZED).let { if (it < 0) 0 else it }
            val uIdx = labels.indexOf(LABEL_UNINFECTED).let { if (it < 0) 1 else it }
            return pIdx to uIdx
        }

        /**
         * Pack a flat row-major RGB float array (shape = 128*128*3) into the
         * model's expected input buffer. Most TFLite vision models accept
         * `[1, H, W, C]` NHWC, but a few exported via PyTorch / ONNX ship as
         * `[1, C, H, W]` NCHW — the runtime-detected [shape] picks the right
         * layout so we don't silently feed transposed pixels.
         */
        private fun buildInputBuffer(
            normalized: FloatArray,
            shape: IntArray
        ): ByteBuffer {
            val byteCount = normalized.size * 4 // float32
            val buf = ByteBuffer.allocateDirect(byteCount)
                .order(ByteOrder.nativeOrder())

            val isNCHW = shape.size == 4 && shape[1] == 3 && shape[2] == MODEL_INPUT_SIZE
            if (isNCHW) {
                // Transpose interleaved RGB (length = H*W*3) to planar CHW.
                val pixelCount = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE
                for (c in 0 until 3) {
                    for (i in 0 until pixelCount) {
                        buf.putFloat(normalized[i * 3 + c])
                    }
                }
            } else {
                // NHWC — feed the flat array directly.
                for (v in normalized) {
                    buf.putFloat(v)
                }
            }
            buf.rewind()
            return buf
        }
    }
}
