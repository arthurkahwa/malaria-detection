package com.malaria.android.services

import android.content.Context
import com.malaria.registry.ModelRegistryEntry
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.coroutines.CoroutineContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * State of a non-bundled model. Mirrors iOS `ModelDownloadState`.
 */
sealed class ModelDownloadState {
    data object NotDownloaded : ModelDownloadState()
    data class Downloading(val progress: Double) : ModelDownloadState() // 0.0–1.0
    data object Downloaded : ModelDownloadState()
    data class Failed(val message: String) : ModelDownloadState()
}

/**
 * Downloads and caches Hugging Face–hosted `.tflite` models.
 *
 * Files are stored in `filesDir/MalariaDetector/models/Malaria_{modelId}.tflite`.
 * The shared-module [com.malaria.ml.Classifier] actual checks that directory
 * before falling back to the bundled asset so the classifier can use a
 * downloaded model without being re-constructed.
 *
 * Unlike the iOS counterpart there is no "compiling" step — TFLite files
 * are used directly after download.
 */
class ModelDownloadService(
    private val context: Context,
    private val scope: CoroutineScope,
) {

    private val modelsDir: File =
        File(context.filesDir, "MalariaDetector/models").also { it.mkdirs() }

    private val _downloadStates = MutableStateFlow<Map<String, ModelDownloadState>>(emptyMap())
    val downloadStates: StateFlow<Map<String, ModelDownloadState>> = _downloadStates.asStateFlow()

    private val activeJobs = mutableMapOf<String, Job>()

    init {
        val existing = modelsDir.listFiles { f -> f.extension == "tflite" } ?: emptyArray()
        if (existing.isNotEmpty()) {
            val initial = existing.associate { f ->
                val id = f.nameWithoutExtension.removePrefix("Malaria_")
                id to ModelDownloadState.Downloaded
            }
            _downloadStates.value = initial
        }
    }

    fun download(entry: ModelRegistryEntry) {
        val modelId = entry.id
        val current = _downloadStates.value[modelId]
        if (current is ModelDownloadState.Downloading || current is ModelDownloadState.Downloaded) return

        val androidPath = entry.androidPath ?: run {
            setState(modelId, ModelDownloadState.Failed("Missing androidPath in registry"))
            return
        }
        val hfRepo = entry.huggingfaceRepo ?: run {
            setState(modelId, ModelDownloadState.Failed("Missing huggingfaceRepo in registry"))
            return
        }

        val repo = hfRepo.replace("{maintainer}", "arthurkahwa")
        val urlString = "https://huggingface.co/$repo/resolve/main/$androidPath"

        setState(modelId, ModelDownloadState.Downloading(0.0))

        val job = scope.launch(Dispatchers.IO) {
            performDownload(modelId, urlString, entry.androidFileSizeMb)
        }
        activeJobs[modelId] = job
    }

    fun deleteModel(modelId: String) {
        activeJobs[modelId]?.cancel()
        activeJobs.remove(modelId)
        modelFile(modelId).delete()
        setState(modelId, ModelDownloadState.NotDownloaded)
    }

    fun cachedModelFile(modelId: String): File? {
        val f = modelFile(modelId)
        return if (f.exists()) f else null
    }

    fun clearAllCaches(settings: SettingsStore) {
        activeJobs.values.forEach { it.cancel() }
        activeJobs.clear()
        modelsDir.listFiles()?.forEach { it.delete() }
        _downloadStates.value = emptyMap()
        scope.launch {
            if (settings.defaultModelId.value != "BNLeaky_Keras") {
                settings.updateDefaultModel("BNLeaky_Keras")
            }
        }
    }

    val hasDownloadedModels: Boolean
        get() = _downloadStates.value.values.any { it is ModelDownloadState.Downloaded }

    val downloadedModelCount: Int
        get() = _downloadStates.value.values.count { it is ModelDownloadState.Downloaded }

    val totalCacheSizeMb: Double
        get() = (modelsDir.listFiles { f -> f.extension == "tflite" } ?: emptyArray())
            .sumOf { it.length() }.toDouble() / (1024.0 * 1024.0)

    // MARK: - Private helpers

    private fun modelFile(modelId: String) = File(modelsDir, "Malaria_$modelId.tflite")

    private fun setState(modelId: String, state: ModelDownloadState) {
        _downloadStates.value = _downloadStates.value + (modelId to state)
    }

    private suspend fun performDownload(
        modelId: String,
        urlString: String,
        fileSizeEstimateMb: Double,
    ) {
        val dest = modelFile(modelId)
        val tmp = File(dest.parent, "${dest.name}.tmp")

        // Capture the coroutine context before entering non-suspend lambdas (use {}).
        val ctx: CoroutineContext = currentCoroutineContext()

        try {
            val conn = URL(urlString).openConnection() as HttpURLConnection
            conn.connectTimeout = 30_000
            conn.readTimeout = 120_000
            conn.connect()

            if (conn.responseCode == 404) {
                withContext(Dispatchers.Main) {
                    setState(modelId, ModelDownloadState.Failed("Model not yet available on Hugging Face"))
                }
                return
            }
            if (conn.responseCode !in 200..299) {
                withContext(Dispatchers.Main) {
                    setState(modelId, ModelDownloadState.Failed("HTTP ${conn.responseCode}"))
                }
                return
            }

            val contentLength = conn.contentLengthLong
            val estimatedBytes =
                if (contentLength > 0) contentLength
                else (fileSizeEstimateMb * 1_048_576).toLong()

            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val buf = ByteArray(256 * 1024)
                    var received = 0L
                    var lastReportedMb = 0L
                    var bytesRead: Int

                    while (input.read(buf).also { bytesRead = it } >= 0) {
                        ctx.ensureActive()
                        output.write(buf, 0, bytesRead)
                        received += bytesRead

                        // Throttle progress updates to every ~0.5 MB.
                        val mbReceived = received / (512 * 1024)
                        if (mbReceived != lastReportedMb && estimatedBytes > 0) {
                            lastReportedMb = mbReceived
                            val progress = (received.toDouble() / estimatedBytes).coerceIn(0.0, 1.0)
                            withContext(Dispatchers.Main) {
                                setState(modelId, ModelDownloadState.Downloading(progress))
                            }
                        }
                    }
                }
            }

            if (dest.exists()) dest.delete()
            tmp.renameTo(dest)

            withContext(Dispatchers.Main) {
                setState(modelId, ModelDownloadState.Downloaded)
            }

        } catch (e: CancellationException) {
            tmp.delete()
            withContext(Dispatchers.Main) {
                setState(modelId, ModelDownloadState.NotDownloaded)
            }
            throw e
        } catch (e: Exception) {
            tmp.delete()
            withContext(Dispatchers.Main) {
                setState(modelId, ModelDownloadState.Failed(e.message ?: "Download failed"))
            }
        }
    }
}
