package com.malaria.android.services

import android.content.Context
import java.security.MessageDigest

/**
 * Computes and caches the SHA-256 hash of a bundled TFLite model asset.
 *
 * The hash is computed once per model ID (stream the asset, feed it through
 * [MessageDigest] in 8 KB chunks) and cached for the lifetime of the process.
 * Falls back to the model ID string if the asset is unavailable.
 *
 * [attachAppContext] must be called from [MalariaApplication.onCreate] before
 * any prediction is recorded.
 */
object ModelHashCache {

    @Volatile
    internal var appContextRef: Context? = null

    fun attachAppContext(context: Context) {
        appContextRef = context.applicationContext
    }

    private val cache = mutableMapOf<String, String>()

    fun hash(modelId: String): String {
        synchronized(cache) { cache[modelId] }?.let { return it }
        val h = computeHash(modelId) ?: modelId
        synchronized(cache) { cache[modelId] = h }
        return h
    }

    private fun computeHash(modelId: String): String? = try {
        val context = appContextRef ?: return null
        val digest = MessageDigest.getInstance("SHA-256")
        context.assets.open("models/Malaria_${modelId}.tflite").use { stream ->
            val buf = ByteArray(8192)
            var read: Int
            while (stream.read(buf).also { read = it } != -1) {
                digest.update(buf, 0, read)
            }
        }
        digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xFF) }
    } catch (_: Exception) {
        null
    }
}
