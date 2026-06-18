package com.malaria.android.services

import android.content.Context
import androidx.security.crypto.EncryptedFile
import androidx.security.crypto.MasterKey
import com.malaria.android.data.entities.AuditAction
import com.malaria.crashlogs.CrashLogRecord
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.json.Json
import java.io.File

/**
 * Settings → Crash logs facade (spec §16).
 *
 * Lists / reads / shares the on-device JSON crash log files written by
 * [CrashLogWriter]. Sweeps logs older than 30 days on every init
 * (spec §16: "auto-expire after 30 days (file modification time check on
 * every app launch)"). Audits each share as `crash_log_shared` with the
 * incident UUID.
 *
 * Mirrors `iosApp/Services/CrashLogStore.swift`. Where iOS uses
 * `NSFileProtectionComplete`, this side uses `EncryptedFile` (Jetpack
 * Security / Tink AES-256-GCM under the Android Keystore).
 */
class CrashLogStore(
    private val context: Context,
    private val auditLog: AuditLog,
    directoryOverride: File? = null,
) {

    private val directory: File = directoryOverride
        ?: File(context.filesDir, CRASHLOGS_SUBDIR)

    private val _entries = MutableStateFlow<List<CrashLogEntry>>(emptyList())
    val entries: StateFlow<List<CrashLogEntry>> = _entries.asStateFlow()

    init {
        directory.mkdirs()
        sweepExpired(directory)
        refresh()
    }

    /** Re-enumerate the directory. Newest first by mtime. */
    fun refresh() {
        _entries.value = enumerate(directory)
    }

    /** Underlying file used by the FileProvider intent. Spec §16: share sheet. */
    fun fileFor(entry: CrashLogEntry): File = entry.file

    /**
     * Decode a crash log on demand. The Settings list view only displays
     * timestamp + incident UUID per spec §16 ("Settings → Crash logs
     * shows a list with timestamp and incident UUID").
     */
    fun read(entry: CrashLogEntry): CrashLogRecord? {
        return try {
            val masterKey = MasterKey.Builder(context, CrashLogWriter.MASTER_KEY_ALIAS)
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .build()
            val encrypted = EncryptedFile.Builder(
                context,
                entry.file,
                masterKey,
                EncryptedFile.FileEncryptionScheme.AES256_GCM_HKDF_4KB,
            ).build()
            val text = encrypted.openFileInput().bufferedReader(Charsets.UTF_8).use { it.readText() }
            Json { ignoreUnknownKeys = true }.decodeFromString(CrashLogRecord.serializer(), text)
        } catch (_: Throwable) {
            null
        }
    }

    /**
     * Spec §16: "Sharing is audited as `CRASH_LOG_SHARED` with the
     * incident UUID."
     *
     * Suspending because [AuditLog.write] is suspending. Settings → Crash
     * logs calls this from a `LaunchedEffect` after the share intent
     * returns.
     */
    suspend fun didShare(entry: CrashLogEntry, actorId: String, actorRole: String) {
        auditLog.write(
            action = AuditAction.CrashLogShared,
            actorId = actorId,
            actorRoleAtTime = actorRole,
            resourceType = "crash_log",
            resourceId = entry.incidentId,
            metadata = emptyMap(),
        )
    }

    /** Convenience for the section header readout. */
    fun count(): Int = _entries.value.size

    companion object {
        /** Spec §16: 30 days. */
        const val MAX_AGE_MILLIS: Long = 30L * 24 * 60 * 60 * 1000

        /** Subdirectory under `context.filesDir`. */
        const val CRASHLOGS_SUBDIR: String = "crashlogs"

        /** Internal helper kept companion-level so tests can invoke without an instance. */
        fun sweepExpired(directory: File) {
            if (!directory.exists()) return
            val cutoff = System.currentTimeMillis() - MAX_AGE_MILLIS
            directory.listFiles { f -> f.isFile && f.extension == "json" }?.forEach { file ->
                if (file.lastModified() < cutoff) {
                    file.delete()
                }
            }
        }

        /** Newest first by mtime. */
        fun enumerate(directory: File): List<CrashLogEntry> {
            val files = directory.listFiles { f -> f.isFile && f.extension == "json" } ?: return emptyList()
            return files.mapNotNull { file ->
                val incidentId = file.nameWithoutExtension
                CrashLogEntry(
                    incidentId = incidentId,
                    timestampMillis = file.lastModified(),
                    sizeBytes = file.length(),
                    file = file,
                )
            }.sortedByDescending { it.timestampMillis }
        }
    }
}

/** Row model for the Compose Crash logs list. */
data class CrashLogEntry(
    val incidentId: String,
    val timestampMillis: Long,
    val sizeBytes: Long,
    val file: File,
)
