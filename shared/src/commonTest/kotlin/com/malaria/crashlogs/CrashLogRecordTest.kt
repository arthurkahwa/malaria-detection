package com.malaria.crashlogs

import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Common-module tests for the [CrashLogRecord] DTO (spec §16).
 *
 * The privacy posture of the crash log relies on the struct shape itself —
 * a third-party reviewer can verify here that no forbidden field surfaces
 * in the JSON.
 */
class CrashLogRecordTest {

    private val json = Json { prettyPrint = false; encodeDefaults = true }

    private fun sample(): CrashLogRecord = CrashLogRecord(
        incidentId = "00000000-0000-0000-0000-000000000001",
        timestampIso8601 = "2026-05-21T12:34:56Z",
        appVersion = "0.1.0",
        osVersion = "iOS 26.0",
        deviceModelClass = "iPhone15,2",
        stackTrace = "Caused by: kotlin.IllegalStateException\n\tat com.malaria.X(Y.kt:1)",
        recentActionTypes = listOf("prediction_created", "override_recorded"),
        memoryPressure = "resident_mb=148",
        deviceUnlocked = true,
    )

    @Test
    fun jsonRoundTrip_preservesAllFields() {
        val original = sample()
        val encoded = json.encodeToString(CrashLogRecord.serializer(), original)
        val decoded = json.decodeFromString(CrashLogRecord.serializer(), encoded)
        assertEquals(original, decoded)
    }

    @Test
    fun encodedJson_doesNotContainForbiddenFields() {
        // Spec §16: the crash log must NOT contain prediction data, override
        // notes, session labels, clinician initials, actor UUIDs, image
        // hashes, clinic config, or consent records. The contract is enforced
        // structurally — these strings should not appear as JSON keys.
        val encoded = json.encodeToString(CrashLogRecord.serializer(), sample())
        val forbiddenKeys = listOf(
            "\"parasitizedProb\"",
            "\"parasitized_prob\"",
            "\"overrideNotes\"",
            "\"override_notes\"",
            "\"sessionLabel\"",
            "\"session_label\"",
            "\"initials\"",
            "\"actorId\"",
            "\"actor_id\"",
            "\"imageHash\"",
            "\"image_hash\"",
            "\"clinicName\"",
            "\"clinic_name\"",
            "\"jurisdiction\"",
            "\"lawfulBasis\"",
            "\"lawful_basis\"",
            "\"consent\"",
        )
        for (key in forbiddenKeys) {
            assertFalse(
                encoded.contains(key),
                "Crash log JSON must not contain forbidden key $key — found in: $encoded",
            )
        }
    }

    @Test
    fun encodedJson_containsAllSpecFields() {
        // The other half of the contract: every spec §16 field IS present.
        val encoded = json.encodeToString(CrashLogRecord.serializer(), sample())
        val required = listOf(
            "incidentId",
            "timestampIso8601",
            "appVersion",
            "osVersion",
            "deviceModelClass",
            "stackTrace",
            "recentActionTypes",
            "memoryPressure",
            "deviceUnlocked",
        )
        for (field in required) {
            assertTrue(
                encoded.contains("\"$field\""),
                "Expected $field in JSON, got: $encoded",
            )
        }
    }
}
