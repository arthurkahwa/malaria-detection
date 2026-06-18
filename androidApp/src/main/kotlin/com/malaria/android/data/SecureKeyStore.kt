package com.malaria.android.data

import android.content.Context
import android.content.pm.PackageManager
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Hardware-backed key store for the SQLCipher database passphrase
 * (spec §19 snippet).
 *
 * **Design.** SQLCipher accepts a `byte[]` passphrase, not a keystore-resident
 * key. The Android Keystore lets us store an AES-256 GCM key under StrongBox
 * when available, but the keystore key can't be exported. So we use it as a
 * key-wrapping key:
 *
 *  1. On first launch, generate a random 32-byte SQLCipher passphrase.
 *  2. Encrypt the passphrase with the Keystore-resident AES-256 GCM key.
 *  3. Persist `iv || ciphertext` (base64) in plain SharedPreferences. The
 *     blob is useless without the keystore key, which never leaves the
 *     secure boundary.
 *  4. On subsequent launches, decrypt the blob and return the passphrase.
 *
 * StrongBox (TEE chip) is requested when the device supports it; software-
 * backed fallback otherwise. Both meet spec §8's "hardware-backed key"
 * intent on modern devices; the StrongBox flag is the strength upgrade.
 */
object SecureKeyStore {

    private const val ANDROID_KEYSTORE = "AndroidKeyStore"
    private const val KEY_ALIAS = "malaria_db_key_v1"
    private const val PREFS_NAME = "malaria_db_keystore"
    private const val PREFS_KEY_CIPHERTEXT = "wrapped_passphrase_b64"
    private const val PREFS_KEY_IV = "wrap_iv_b64"

    private const val PASSPHRASE_BYTES = 32 // SQLCipher AES-256
    private const val GCM_TAG_BITS = 128

    /**
     * Returns the SQLCipher passphrase, creating one if needed. Subsequent
     * calls return the same passphrase.
     *
     * Note: deliberately NOT setting `setUserAuthenticationRequired(true)` on
     * the wrapping key. The spec §19 snippet shows that flag, but pairing it
     * with `setUserAuthenticationParameters(300, AUTH_BIOMETRIC_STRONG)` means
     * the database can only be opened after a fresh biometric prompt —
     * incompatible with Room's eager init at `MainActivity.onCreate` (the
     * BiometricPrompt happens later, from a composable). Phase 6's full
     * auth integration revisits this; v1 ships with the keystore as the
     * encryption boundary and biometric guarding the app UI layer.
     */
    fun getOrCreateDatabaseKey(context: Context): ByteArray {
        ensureWrappingKey(context)
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        val wrappedB64 = prefs.getString(PREFS_KEY_CIPHERTEXT, null)
        val ivB64 = prefs.getString(PREFS_KEY_IV, null)
        if (wrappedB64 != null && ivB64 != null) {
            val wrapped = Base64.decode(wrappedB64, Base64.NO_WRAP)
            val iv = Base64.decode(ivB64, Base64.NO_WRAP)
            return unwrap(wrapped, iv)
        }

        // First-launch path: mint a fresh passphrase, wrap it, persist
        // the ciphertext + IV.
        val passphrase = ByteArray(PASSPHRASE_BYTES).also { SecureRandom().nextBytes(it) }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, wrappingKey())
        val ciphertext = cipher.doFinal(passphrase)
        val iv = cipher.iv

        prefs.edit()
            .putString(PREFS_KEY_CIPHERTEXT, Base64.encodeToString(ciphertext, Base64.NO_WRAP))
            .putString(PREFS_KEY_IV, Base64.encodeToString(iv, Base64.NO_WRAP))
            .apply()

        return passphrase
    }

    private fun ensureWrappingKey(context: Context) {
        val keystore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        if (keystore.containsAlias(KEY_ALIAS)) return

        val builder = KeyGenParameterSpec.Builder(
            KEY_ALIAS,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
        )
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .setRandomizedEncryptionRequired(true)

        if (context.packageManager.hasSystemFeature(
                PackageManager.FEATURE_STRONGBOX_KEYSTORE,
            )
        ) {
            try {
                builder.setIsStrongBoxBacked(true)
            } catch (_: Throwable) {
                // Some emulators advertise StrongBox but reject key creation;
                // fall back to TEE-backed software-only keystore.
            }
        }

        val keyGen = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES,
            ANDROID_KEYSTORE,
        )
        try {
            keyGen.init(builder.build())
            keyGen.generateKey()
        } catch (_: Throwable) {
            // Retry without StrongBox (some devices fail key creation
            // even when the feature is reported).
            keyGen.init(
                KeyGenParameterSpec.Builder(
                    KEY_ALIAS,
                    KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
                )
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setKeySize(256)
                    .setRandomizedEncryptionRequired(true)
                    .build(),
            )
            keyGen.generateKey()
        }
    }

    private fun wrappingKey(): SecretKey {
        val keystore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        return keystore.getKey(KEY_ALIAS, null) as SecretKey
    }

    private fun unwrap(ciphertext: ByteArray, iv: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, wrappingKey(), GCMParameterSpec(GCM_TAG_BITS, iv))
        return cipher.doFinal(ciphertext)
    }
}
