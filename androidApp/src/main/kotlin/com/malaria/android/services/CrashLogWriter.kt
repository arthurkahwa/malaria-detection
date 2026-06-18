package com.malaria.android.services

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import androidx.security.crypto.EncryptedFile
import androidx.security.crypto.MasterKey
import com.malaria.crashlogs.CrashLogRecord
import com.malaria.crashlogs.RecentActionRing
import kotlinx.serialization.json.Json
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID

/**
 * Installs the JVM uncaught-exception handler that writes a crash log per
 * spec §16.
 *
 * Pragmatic Phase 14 v0.1 scope (see README "Known limitations"):
 * - Catches JVM exceptions via [Thread.setDefaultUncaughtExceptionHandler]
 *   only. NDK-native signal-handler capture is deferred to Phase 15
 *   (would require shipping a native library).
 * - Writer uses `androidx.security.crypto.EncryptedFile` (Jetpack Security)
 *   for at-rest encryption, sharing the Keystore master key alias pattern
 *   with the Phase 5 SQLCipher passphrase wrap. The strictly signal-safe
 *   POSIX `open()` / `write()` / `close()` path is deferred to Phase 15.
 *
 * The environment (`appVersion`, `osVersion`, `deviceModelClass`, …) is
 * captured once at install time so the handler doesn't have to touch
 * `BuildConfig` / `Build` during the crash path.
 */
object CrashLogWriter {

    /**
     * Snapshot captured at install time. The crash handler reads the
     * lambdas/values from here without re-querying the OS.
     */
    data class Environment(
        val directory: File,
        val appVersion: String,
        val osVersion: String,
        val deviceModelClass: String,
        /**
         * Best-effort "is the device unlocked?" — passes through
         * `Context.getSystemService(KeyguardManager).isDeviceLocked`. Read
         * at crash time from the supplied lambda so the most recent
         * answer wins.
         */
        val isUnlocked: () -> Boolean,
        val activityManager: ActivityManager?,
    )

    @Volatile
    private var environment: Environment? = null

    @Volatile
    private var previousHandler: Thread.UncaughtExceptionHandler? = null

    /**
     * Install the global handler. Idempotent — calling again replaces the
     * stored [environment] but only swaps the [Thread] handler the first
     * time.
     */
    fun install(env: Environment) {
        environment = env
        env.directory.mkdirs()
        if (previousHandler != null) return
        previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                val captured = environment
                if (captured != null) {
                    val record = makeRecord(captured, throwable)
                    writeRecord(captured.directory, record)
                }
            } catch (_: Throwable) {
                // Swallow — we MUST NOT swallow the original crash. Fall
                // through to the previous handler so the process still dies
                // normally and the OS gets its crash report.
            }
            previousHandler?.uncaughtException(thread, throwable)
        }

        // Install NDK signal handler for native crashes (SIGSEGV, SIGBUS, etc.)
        // from TFLite native libs. Wrapped so that a missing native library
        // (e.g. in unit-test JVM or stripped APK) does not prevent the JVM
        // handler above from working.
        try {
            NativeCrashHandler.install(env.directory.absolutePath)
        } catch (_: Throwable) {
            // Native library unavailable — proceed without native crash capture.
        }
    }

    /**
     * Synthesise a [CrashLogRecord] from the current environment and the
     * crashing [throwable]. Exposed so tests can exercise the shape
     * without actually crashing.
     */
    fun makeRecord(env: Environment, throwable: Throwable): CrashLogRecord {
        val sw = StringWriter()
        throwable.printStackTrace(PrintWriter(sw))
        return CrashLogRecord(
            incidentId = UUID.randomUUID().toString(),
            timestampIso8601 = isoNow(),
            appVersion = env.appVersion,
            osVersion = env.osVersion,
            deviceModelClass = env.deviceModelClass,
            stackTrace = sw.toString(),
            recentActionTypes = RecentActionRing.shared.snapshot(),
            memoryPressure = memoryPressureString(env.activityManager),
            deviceUnlocked = runCatching { env.isUnlocked() }.getOrDefault(false),
        )
    }

    /**
     * Persist a [CrashLogRecord] under [directory]/{incidentId}.json via
     * `EncryptedFile`. Returns the file on success, null on failure.
     *
     * `EncryptedFile` uses Tink AES-256-GCM under the hood, with the
     * `MasterKey` AES-256 wrapping key stored in the Android Keystore
     * under the alias [MASTER_KEY_ALIAS]. The alias is the same pattern
     * Phase 5 uses for the SQLCipher passphrase wrap; the master key
     * never leaves the Keystore.
     */
    fun writeRecord(directory: File, record: CrashLogRecord): File? {
        return try {
            directory.mkdirs()
            val context = appContextRef ?: return null
            val masterKey = MasterKey.Builder(context, MASTER_KEY_ALIAS)
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .build()
            val file = File(directory, "${record.incidentId}.json")
            // `EncryptedFile.openFileOutput()` requires the target file
            // not to exist — remove any stale file first (won't happen in
            // practice since incidentId is a fresh UUID).
            if (file.exists()) file.delete()
            val encrypted = EncryptedFile.Builder(
                context,
                file,
                masterKey,
                EncryptedFile.FileEncryptionScheme.AES256_GCM_HKDF_4KB,
            ).build()
            val json = Json { encodeDefaults = true }.encodeToString(
                CrashLogRecord.serializer(),
                record,
            ).toByteArray(Charsets.UTF_8)
            encrypted.openFileOutput().use { it.write(json) }
            file
        } catch (_: Throwable) {
            null
        }
    }

    /**
     * Hook called from `MainActivity.onCreate` so `EncryptedFile` can be
     * constructed from inside the static handler. Kept package-private
     * via Kotlin's default visibility (`public` for the object's
     * `install` method, internal for the context plumbing).
     */
    @Volatile
    internal var appContextRef: Context? = null

    /** Compose-time wiring helper. Avoid leaking Activities; pass `applicationContext`. */
    fun attachAppContext(context: Context) {
        appContextRef = context.applicationContext
    }

    // MARK: - Helpers

    private fun isoNow(): String {
        val fmt = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        fmt.timeZone = TimeZone.getTimeZone("UTC")
        return fmt.format(Date())
    }

    /**
     * `resident_mb=…` from [ActivityManager.MemoryInfo]. Cheap, allocation-
     * light, and stays within `android.app` rather than reaching for
     * `Debug.MemoryInfo` (which forks a binder call). Spec §16 only asks
     * for "memory pressure"; the resident-size readout is a reasonable
     * proxy on both platforms.
     */
    fun memoryPressureString(activityManager: ActivityManager?): String {
        if (activityManager == null) return "unknown"
        val info = ActivityManager.MemoryInfo()
        return try {
            activityManager.getMemoryInfo(info)
            val availableMb = info.availMem / (1024 * 1024)
            val totalMb = info.totalMem / (1024 * 1024)
            val lowMemory = info.lowMemory
            "available_mb=$availableMb,total_mb=$totalMb,low=$lowMemory"
        } catch (_: Throwable) {
            "unknown"
        }
    }

    /**
     * Compact "manufacturer / model" string (spec §16: e.g. "Pixel 9 Pro").
     */
    fun currentDeviceModelClass(): String {
        return "${Build.MANUFACTURER} ${Build.MODEL}".trim()
    }

    /**
     * Single Keystore alias for the EncryptedFile master key. Distinct
     * from the SQLCipher wrap alias (`malaria_db_key_v1`) so that
     * rotating one doesn't invalidate the other; same Keystore-master-key
     * pattern though.
     */
    const val MASTER_KEY_ALIAS: String = "malaria_crashlogs_master_key_v1"
}
