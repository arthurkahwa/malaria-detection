package com.malaria.export

import org.kotlincrypto.hash.sha2.SHA256
import org.kotlincrypto.macs.hmac.sha2.HmacSHA256

/**
 * Computes the HMAC-SHA256 signature that seals an export bundle (spec §14).
 *
 * ## Key derivation
 *
 * The signing key is derived from the device UUID and an export-time
 * timestamp salt:
 *
 * ```
 *   key = SHA-256( deviceUuid + ":" + timestampSalt )    // 32 bytes
 * ```
 *
 * The key is then used as the HMAC-SHA256 key over the unsigned bundle JSON.
 * The signature is emitted as lowercase hex.
 *
 * ## Reproducibility
 *
 * Anyone with the original device's UUID and the export's `exportTimestamp`
 * field can re-derive the key and verify the bundle. This is *not* an
 * external notarization — for that, a deployer would replace the key
 * derivation with a clinic-supplied private signing key in v2. The intent in
 * v1 is integrity verification, not third-party trust.
 *
 * ## Determinism
 *
 * The same `(unsignedJson, deviceUuid, timestampSalt)` triple always produces
 * the same hex signature on both iOS and Android — `org.kotlincrypto.macs`
 * is a pure-Kotlin multiplatform implementation with no platform-specific
 * dispatch, so this is a static guarantee (see `ExportSignerTest`).
 */
object ExportSigner {

    /**
     * Sign the canonical [unsignedJson] form. Returns the lowercase
     * hexadecimal HMAC-SHA256 of [unsignedJson] under the key derived from
     * [deviceUuid] and [timestampSalt].
     */
    fun sign(unsignedJson: String, deviceUuid: String, timestampSalt: String): String {
        val key = deriveKey(deviceUuid, timestampSalt)
        val mac = HmacSHA256(key)
        val tag = mac.doFinal(unsignedJson.encodeToByteArray())
        return tag.toLowercaseHex()
    }

    /**
     * Exposed for tests and any future verifier path. The derivation is
     * trivial — `SHA-256( deviceUuid + ":" + timestampSalt )` — but
     * publishing the helper avoids duplication on the verify side.
     */
    fun deriveKey(deviceUuid: String, timestampSalt: String): ByteArray {
        val material = (deviceUuid + ":" + timestampSalt).encodeToByteArray()
        return SHA256().digest(material)
    }
}

/** Lowercase hex encoding for a HMAC tag (matches `Preprocessor.sha256Hex`). */
private fun ByteArray.toLowercaseHex(): String {
    val hex = StringBuilder(size * 2)
    for (b in this) {
        val v = b.toInt() and 0xFF
        hex.append(HEX_CHARS[v ushr 4])
        hex.append(HEX_CHARS[v and 0x0F])
    }
    return hex.toString()
}

private val HEX_CHARS = "0123456789abcdef".toCharArray()
