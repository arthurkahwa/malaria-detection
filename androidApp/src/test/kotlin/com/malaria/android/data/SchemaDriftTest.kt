package com.malaria.android.data

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Schema-drift test (spec §8 / §20). Compares Room's exported schema JSON
 * against a hardcoded canonical field set derived from `docs/SCHEMA.md`.
 *
 * **Why hardcoded.** Parsing `docs/SCHEMA.md` as authoritative was rejected
 * — the markdown structure is meant for humans, the canonical entity layout
 * is small enough to inline, and a divergence between the hardcoded
 * expected lists below and `docs/SCHEMA.md` is itself a doc-vs-code drift
 * that a human reviewer catches at PR time. The drift this test fights is
 * schema-vs-code: if a developer renames a Room column without updating
 * `docs/SCHEMA.md`, the test fails here and the doc update is forced
 * back into the same PR.
 *
 * Requires the database to have been built at least once so the schema
 * JSON exists. `assembleDebug` triggers Room's `copyRoomSchemas` task; CI
 * runs that before this test (see `KMP_App_Specification.md` §21).
 */
class SchemaDriftTest {

    private val schemaFile = File(
        "schemas/com.malaria.android.data.MalariaDatabase/1.json",
    )

    @Test
    fun `schema export file exists`() {
        assertTrue(
            "Run `./gradlew :androidApp:assembleDebug` first so Room exports the schema",
            schemaFile.exists(),
        )
    }

    @Test
    fun `predictions table fields match canonical schema`() {
        assertTableFields(
            tableName = "predictions",
            expected = setOf(
                "id",
                "session_id",
                "timestamp",
                "model_id",
                "model_version",
                "parasitized_prob",
                "uninfected_prob",
                "label",
                "threshold",
                "flagged_for_review",
                "inference_ms",
                "image_hash",
                "clinician_override",
                "override_context",
                "duplicate_of_id",
                "session_label",
                "app_version",
                "os_version",
            ),
        )
    }

    @Test
    fun `audit_entries table fields match canonical schema`() {
        assertTableFields(
            tableName = "audit_entries",
            expected = setOf(
                "id",
                "seq",
                "timestamp",
                "actor_id",
                "actor_role_at_time",
                "action",
                "resource_type",
                "resource_id",
                "metadata_json",
                "override_context",
                "override_reason",
                "override_notes",
                "context_reviewed",
                "override_actor_initials",
                "app_version",
                "os_version",
            ),
        )
    }

    @Test
    fun `clinician_profiles table fields match canonical schema`() {
        assertTableFields(
            tableName = "clinician_profiles",
            expected = setOf(
                "id",
                "actor_id",
                "role",
                "initials",
                "enrolled_at",
                "biometric_enrolled",
            ),
        )
    }

    @Test
    fun `consent_records table fields match canonical schema`() {
        assertTableFields(
            tableName = "consent_records",
            expected = setOf(
                "id",
                "timestamp",
                "actor_id",
                "consent_type",
                "document_version",
                "value",
                "app_version",
            ),
        )
    }

    @Test
    fun `database version is 1`() {
        val db = JSONObject(schemaFile.readText()).getJSONObject("database")
        assertEquals(1, db.getInt("version"))
    }

    private fun assertTableFields(tableName: String, expected: Set<String>) {
        val entity = entityByTable(tableName)
        val actual = entity.getJSONArray("fields").columnNames()
        assertEquals("Drift in `$tableName` columns", expected, actual)
    }

    private fun entityByTable(tableName: String): JSONObject {
        val root = JSONObject(schemaFile.readText())
        val entities = root.getJSONObject("database").getJSONArray("entities")
        for (i in 0 until entities.length()) {
            val entity = entities.getJSONObject(i)
            if (entity.getString("tableName") == tableName) return entity
        }
        error("No entity for table $tableName in exported schema")
    }

    private fun JSONArray.columnNames(): Set<String> {
        val out = mutableSetOf<String>()
        for (i in 0 until length()) {
            out += getJSONObject(i).getString("columnName")
        }
        return out
    }
}
