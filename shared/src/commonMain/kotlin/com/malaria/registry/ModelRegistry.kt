package com.malaria.registry

import kotlinx.serialization.json.Json

/**
 * Parses `model_registry.json` (passed in as a string) and exposes lookups.
 *
 * This class is pure Kotlin: it does no resource I/O. Each platform reads the
 * JSON from its bundle (iOS) or assets (Android) and passes the string to
 * [parse]. Holding [ModelRegistry] as an injected service after construction
 * is the recommended pattern.
 */
class ModelRegistry private constructor(
    private val manifest: ModelRegistryManifest
) {
    /** Schema version of the parsed manifest. */
    val schemaVersion: String get() = manifest.schemaVersion

    /** All model entries, in registry order. */
    fun all(): List<ModelRegistryEntry> = manifest.models

    /** Returns the entry whose [ModelRegistryEntry.id] matches [id], or null. */
    fun byId(id: String): ModelRegistryEntry? = manifest.models.firstOrNull { it.id == id }

    companion object {
        private val json = Json {
            ignoreUnknownKeys = false
            isLenient = false
        }

        /**
         * Parses [jsonString] into a [ModelRegistry].
         *
         * Throws [kotlinx.serialization.SerializationException] when JSON is
         * malformed or a required field is missing.
         */
        fun parse(jsonString: String): ModelRegistry {
            val manifest = json.decodeFromString(ModelRegistryManifest.serializer(), jsonString)
            return ModelRegistry(manifest)
        }
    }
}
