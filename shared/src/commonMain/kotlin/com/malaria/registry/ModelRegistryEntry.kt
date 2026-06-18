package com.malaria.registry

import kotlinx.serialization.Serializable

/**
 * A single model entry from `model_registry.json`.
 *
 * Nullable fields are populated only when [bundled] is false (a remote model
 * requires download metadata; a bundled model carries it implicitly in the
 * native asset path).
 */
@Serializable
data class ModelRegistryEntry(
    val id: String,
    val displayName: String,
    val framework: String,
    val architecture: String,
    val filenameStem: String,
    val inputSize: Int,
    val paramCount: Long,
    val testAccuracy: Double,
    val bundled: Boolean,
    val huggingfaceRepo: String?,
    val iosPath: String?,
    val androidPath: String?,
    val iosExpectedSha256: String?,
    val androidExpectedSha256: String?,
    val iosFileSizeMb: Double,
    val androidFileSizeMb: Double,
    val description: String
)

/** Top-level shape of `model_registry.json`. */
@Serializable
data class ModelRegistryManifest(
    val schemaVersion: String,
    val models: List<ModelRegistryEntry>
)
