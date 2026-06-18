package com.malaria.domain

/**
 * Raw image data passed to a [com.malaria.ml.Classifier].
 *
 * Bytes are expected to be tightly-packed 8-bit-per-channel RGB
 * (three bytes per pixel, no padding), row-major, top-left origin.
 * Length must equal `width * height * 3`.
 *
 * `equals` and `hashCode` are overridden because [rgbBytes] is a [ByteArray];
 * default data-class equality would compare by reference.
 */
data class ImageInput(
    val rgbBytes: ByteArray,
    val width: Int,
    val height: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ImageInput) return false
        if (width != other.width) return false
        if (height != other.height) return false
        if (!rgbBytes.contentEquals(other.rgbBytes)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = rgbBytes.contentHashCode()
        result = 31 * result + width
        result = 31 * result + height
        return result
    }
}
