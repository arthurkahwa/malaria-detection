package com.malaria.android.services

import android.content.Context
import com.malaria.registry.ModelRegistry
import kotlinx.coroutines.CoroutineScope

/**
 * Loads `model_registry.json` from the Android `assets/` directory and hands
 * it to the shared-module parser.
 *
 * Spec §5 places the canonical registry at
 * `shared/src/commonMain/resources/model_registry.json`; on Android the
 * easiest hand-off is a build-time copy into `assets/`. A Gradle task to
 * automate that copy is a future-phase deliverable — for now the assets file
 * is a hand-maintained placeholder.
 */
class ModelRegistryService(
    private val context: Context,
    @Suppress("unused") private val scope: CoroutineScope,
) {
    suspend fun registry(): ModelRegistry {
        val json = context.assets.open("model_registry.json")
            .bufferedReader()
            .use { it.readText() }
        return ModelRegistry.parse(json)
    }
}
