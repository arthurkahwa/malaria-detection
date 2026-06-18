package com.malaria.preprocessing

import com.malaria.domain.ImageInput
import org.kotlincrypto.hash.sha2.SHA256

/**
 * Pure-Kotlin image preprocessing.
 *
 * Both platforms call this from their `actual class Classifier` so that the
 * resize and normalization are bit-for-bit identical across iOS and Android.
 */

/**
 * Resizes [input] to [targetSize] x [targetSize] using nearest-neighbor
 * sampling and normalizes RGB byte channels into the closed interval `[0, 1]`.
 *
 * The returned [FloatArray] has length `targetSize * targetSize * 3` and is
 * laid out row-major, top-left origin, three interleaved channels per pixel:
 * `[r0, g0, b0, r1, g1, b1, ...]`.
 *
 * Nearest-neighbor was chosen over bilinear or bicubic to keep the
 * implementation pure-Kotlin and free of floating-point platform divergence;
 * the spec does not mandate a specific resize algorithm.
 *
 * @throws IllegalArgumentException if [input]'s dimensions or byte count are
 *   inconsistent, or if [targetSize] is not positive.
 */
fun preprocess(input: ImageInput, targetSize: Int = 128): FloatArray {
    require(targetSize > 0) { "targetSize must be positive, was $targetSize" }
    require(input.width > 0 && input.height > 0) {
        "ImageInput must have positive width and height (got ${input.width}x${input.height})"
    }
    val expectedBytes = input.width * input.height * 3
    require(input.rgbBytes.size == expectedBytes) {
        "ImageInput.rgbBytes size ${input.rgbBytes.size} does not match width*height*3 = $expectedBytes"
    }

    val out = FloatArray(targetSize * targetSize * 3)
    val srcW = input.width
    val srcH = input.height
    val src = input.rgbBytes

    // Nearest-neighbor sampling. For destination pixel (dx, dy), pick the
    // source pixel whose center is closest in the normalized coordinate space.
    var di = 0
    for (dy in 0 until targetSize) {
        val sy = (dy.toLong() * srcH / targetSize).toInt().coerceIn(0, srcH - 1)
        for (dx in 0 until targetSize) {
            val sx = (dx.toLong() * srcW / targetSize).toInt().coerceIn(0, srcW - 1)
            val si = (sy * srcW + sx) * 3
            // Bytes are signed in Kotlin; mask back to unsigned 0..255 before
            // normalizing.
            out[di] = (src[si].toInt() and 0xFF) / 255f
            out[di + 1] = (src[si + 1].toInt() and 0xFF) / 255f
            out[di + 2] = (src[si + 2].toInt() and 0xFF) / 255f
            di += 3
        }
    }
    return out
}

/**
 * Resizes [input] to [targetSize] x [targetSize] using nearest-neighbor
 * sampling **without normalisation** — channel values stay in `[0, 255]`.
 *
 * Used by the iOS Core ML pipeline, which expects a `CVPixelBuffer` of
 * uint8 BGRA bytes and applies its own internal normalisation as the
 * first op of the model graph. The Android TFLite pipeline can use
 * [preprocess] for its float-tensor input instead.
 *
 * Layout matches [preprocess]: row-major, top-left origin, interleaved
 * RGB per pixel.
 */
fun resizeRGB(input: ImageInput, targetSize: Int = 128): ByteArray {
    require(targetSize > 0) { "targetSize must be positive, was $targetSize" }
    require(input.width > 0 && input.height > 0) {
        "ImageInput must have positive width and height (got ${input.width}x${input.height})"
    }
    val expectedBytes = input.width * input.height * 3
    require(input.rgbBytes.size == expectedBytes) {
        "ImageInput.rgbBytes size ${input.rgbBytes.size} does not match width*height*3 = $expectedBytes"
    }

    val out = ByteArray(targetSize * targetSize * 3)
    val srcW = input.width
    val srcH = input.height
    val src = input.rgbBytes

    var di = 0
    for (dy in 0 until targetSize) {
        val sy = (dy.toLong() * srcH / targetSize).toInt().coerceIn(0, srcH - 1)
        for (dx in 0 until targetSize) {
            val sx = (dx.toLong() * srcW / targetSize).toInt().coerceIn(0, srcW - 1)
            val si = (sy * srcW + sx) * 3
            out[di] = src[si]
            out[di + 1] = src[si + 1]
            out[di + 2] = src[si + 2]
            di += 3
        }
    }
    return out
}

/**
 * Returns the lowercase hexadecimal SHA-256 digest of [bytes].
 *
 * Used to fill the `imageHash` field on [com.malaria.domain.Prediction].
 */
fun sha256Hex(bytes: ByteArray): String {
    val digest = SHA256().digest(bytes)
    return buildString(digest.size * 2) {
        for (b in digest) {
            val v = b.toInt() and 0xFF
            append(HEX_DIGITS[v ushr 4])
            append(HEX_DIGITS[v and 0x0F])
        }
    }
}

private val HEX_DIGITS = "0123456789abcdef".toCharArray()
