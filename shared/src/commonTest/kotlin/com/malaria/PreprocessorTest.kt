package com.malaria

import com.malaria.domain.ImageInput
import com.malaria.preprocessing.preprocess
import com.malaria.preprocessing.sha256Hex
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class PreprocessorTest {

    @Test
    fun preprocessProducesCorrectlySizedOutput() {
        // 4x4 RGB image: 48 bytes.
        val bytes = ByteArray(4 * 4 * 3) { 0 }
        val out = preprocess(ImageInput(bytes, 4, 4), targetSize = 8)
        assertEquals(8 * 8 * 3, out.size)
    }

    @Test
    fun preprocessNormalizesByteToFloatRange() {
        // Single 1x1 white pixel.
        val bytes = byteArrayOf(0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte())
        val out = preprocess(ImageInput(bytes, 1, 1), targetSize = 1)
        assertEquals(3, out.size)
        assertEquals(1.0f, out[0])
        assertEquals(1.0f, out[1])
        assertEquals(1.0f, out[2])
    }

    @Test
    fun preprocessBlackPixelIsZero() {
        val bytes = byteArrayOf(0, 0, 0)
        val out = preprocess(ImageInput(bytes, 1, 1), targetSize = 1)
        assertEquals(0.0f, out[0])
        assertEquals(0.0f, out[1])
        assertEquals(0.0f, out[2])
    }

    @Test
    fun preprocessMidGrayIsHalfRoughly() {
        val bytes = byteArrayOf(0x80.toByte(), 0x80.toByte(), 0x80.toByte())
        val out = preprocess(ImageInput(bytes, 1, 1), targetSize = 1)
        // 0x80 == 128; 128 / 255 ~ 0.5019608
        val expected = 128f / 255f
        assertEquals(expected, out[0])
        assertEquals(expected, out[1])
        assertEquals(expected, out[2])
    }

    @Test
    fun preprocessPreservesRgbChannelsOnFixedTwoByTwo() {
        // 2x2 image, distinct pixels:
        // (0,0) red, (1,0) green, (0,1) blue, (1,1) white.
        val bytes = byteArrayOf(
            0xFF.toByte(), 0, 0,
            0, 0xFF.toByte(), 0,
            0, 0, 0xFF.toByte(),
            0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte()
        )
        // Resize to 2x2 — nearest-neighbor should preserve the four pixels.
        val out = preprocess(ImageInput(bytes, 2, 2), targetSize = 2)
        assertEquals(2 * 2 * 3, out.size)
        // (0,0) red
        assertEquals(1.0f, out[0]); assertEquals(0.0f, out[1]); assertEquals(0.0f, out[2])
        // (1,0) green
        assertEquals(0.0f, out[3]); assertEquals(1.0f, out[4]); assertEquals(0.0f, out[5])
        // (0,1) blue
        assertEquals(0.0f, out[6]); assertEquals(0.0f, out[7]); assertEquals(1.0f, out[8])
        // (1,1) white
        assertEquals(1.0f, out[9]); assertEquals(1.0f, out[10]); assertEquals(1.0f, out[11])
    }

    @Test
    fun preprocessUpscaleIsDeterministic() {
        val bytes = byteArrayOf(0xFF.toByte(), 0, 0, 0, 0xFF.toByte(), 0, 0, 0, 0xFF.toByte(), 0x40, 0x40, 0x40)
        val a = preprocess(ImageInput(bytes, 2, 2), targetSize = 16)
        val b = preprocess(ImageInput(bytes, 2, 2), targetSize = 16)
        assertTrue(a.contentEquals(b))
    }

    @Test
    fun preprocessRejectsMismatchedByteCount() {
        // 2x2 should be 12 bytes; supply 10.
        val bytes = ByteArray(10) { 0 }
        assertFailsWith<IllegalArgumentException> {
            preprocess(ImageInput(bytes, 2, 2), targetSize = 4)
        }
    }

    @Test
    fun preprocessRejectsZeroDimensions() {
        assertFailsWith<IllegalArgumentException> {
            preprocess(ImageInput(ByteArray(0), 0, 0), targetSize = 4)
        }
    }

    @Test
    fun preprocessRejectsNonPositiveTarget() {
        val bytes = ByteArray(3) { 0 }
        assertFailsWith<IllegalArgumentException> {
            preprocess(ImageInput(bytes, 1, 1), targetSize = 0)
        }
    }

    @Test
    fun sha256HexIsDeterministic() {
        val bytes = byteArrayOf(1, 2, 3, 4, 5)
        val h1 = sha256Hex(bytes)
        val h2 = sha256Hex(bytes.copyOf())
        assertEquals(h1, h2)
        assertEquals(64, h1.length)
        assertTrue(h1.all { it in '0'..'9' || it in 'a'..'f' })
    }

    @Test
    fun sha256HexDistinguishesDifferentInputs() {
        val a = sha256Hex(byteArrayOf(1, 2, 3))
        val b = sha256Hex(byteArrayOf(1, 2, 4))
        assertNotEquals(a, b)
    }

    @Test
    fun sha256HexOfEmptyMatchesKnownValue() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assertEquals(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            sha256Hex(ByteArray(0))
        )
    }

    @Test
    fun sha256HexOfAbcMatchesKnownValue() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        assertEquals(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            sha256Hex("abc".encodeToByteArray())
        )
    }
}
