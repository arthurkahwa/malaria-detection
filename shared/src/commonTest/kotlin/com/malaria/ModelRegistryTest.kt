package com.malaria

import com.malaria.registry.ModelRegistry
import kotlinx.serialization.SerializationException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ModelRegistryTest {

    private val validJson = """
        {
          "schemaVersion": "1.0",
          "models": [
            {
              "id": "BNLeaky_Keras",
              "displayName": "BN + LeakyReLU (Keras)",
              "framework": "Keras",
              "architecture": "BN+LeakyReLU",
              "filenameStem": "Malaria_BNLeaky_Keras",
              "inputSize": 128,
              "paramCount": 120000,
              "testAccuracy": 0.97,
              "bundled": true,
              "huggingfaceRepo": null,
              "iosPath": null,
              "androidPath": null,
              "iosExpectedSha256": null,
              "androidExpectedSha256": null,
              "iosFileSizeMb": 16.2,
              "androidFileSizeMb": 5.4,
              "description": "Bundled."
            },
            {
              "id": "EfficientNetB3_Keras",
              "displayName": "EfficientNetB3 (Keras)",
              "framework": "Keras",
              "architecture": "EfficientNetB3",
              "filenameStem": "Malaria_EfficientNetB3_Keras",
              "inputSize": 128,
              "paramCount": 12000000,
              "testAccuracy": 0.985,
              "bundled": false,
              "huggingfaceRepo": "{maintainer}/malaria-detector-models",
              "iosPath": "Malaria_EfficientNetB3_Keras.mlpackage",
              "androidPath": "Malaria_EfficientNetB3_Keras.tflite",
              "iosExpectedSha256": "PENDING_PHASE_MINUS_1",
              "androidExpectedSha256": "PENDING_PHASE_MINUS_1",
              "iosFileSizeMb": 50.4,
              "androidFileSizeMb": 14.8,
              "description": "Best accuracy."
            }
          ]
        }
    """.trimIndent()

    @Test
    fun parsesValidManifest() {
        val registry = ModelRegistry.parse(validJson)
        assertEquals("1.0", registry.schemaVersion)
        assertEquals(2, registry.all().size)
    }

    @Test
    fun byIdReturnsEntryWhenPresent() {
        val registry = ModelRegistry.parse(validJson)
        val entry = registry.byId("EfficientNetB3_Keras")
        assertNotNull(entry)
        assertEquals("EfficientNetB3", entry.architecture)
        assertEquals(false, entry.bundled)
        assertEquals("PENDING_PHASE_MINUS_1", entry.iosExpectedSha256)
    }

    @Test
    fun byIdReturnsNullWhenAbsent() {
        val registry = ModelRegistry.parse(validJson)
        assertNull(registry.byId("does-not-exist"))
    }

    @Test
    fun bundledEntryHasNullDownloadFields() {
        val registry = ModelRegistry.parse(validJson)
        val bundled = registry.byId("BNLeaky_Keras")
        assertNotNull(bundled)
        assertTrue(bundled.bundled)
        assertNull(bundled.huggingfaceRepo)
        assertNull(bundled.iosPath)
        assertNull(bundled.androidPath)
        assertNull(bundled.iosExpectedSha256)
        assertNull(bundled.androidExpectedSha256)
    }

    @Test
    fun malformedJsonFailsToParse() {
        val notJson = "this is not json {"
        assertFailsWith<SerializationException> {
            ModelRegistry.parse(notJson)
        }
    }

    @Test
    fun missingRequiredFieldFailsToParse() {
        // `displayName` is required (non-nullable).
        val missingField = """
            {
              "schemaVersion": "1.0",
              "models": [
                {
                  "id": "X",
                  "framework": "Keras",
                  "architecture": "X",
                  "filenameStem": "X",
                  "inputSize": 128,
                  "paramCount": 1,
                  "testAccuracy": 0.5,
                  "bundled": true,
                  "huggingfaceRepo": null,
                  "iosPath": null,
                  "androidPath": null,
                  "iosExpectedSha256": null,
                  "androidExpectedSha256": null,
                  "iosFileSizeMb": 1.0,
                  "androidFileSizeMb": 1.0,
                  "description": "X"
                }
              ]
            }
        """.trimIndent()
        assertFailsWith<SerializationException> {
            ModelRegistry.parse(missingField)
        }
    }

    @Test
    fun unknownFieldFailsToParse() {
        // Strict parsing helps surface schema drift.
        val unknownField = """
            {
              "schemaVersion": "1.0",
              "unexpectedTopLevel": 42,
              "models": []
            }
        """.trimIndent()
        assertFailsWith<SerializationException> {
            ModelRegistry.parse(unknownField)
        }
    }
}
