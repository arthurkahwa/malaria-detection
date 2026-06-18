package com.malaria.android

import android.app.Application
import com.malaria.android.services.ModelHashCache
import com.malaria.android.services.NativeCrashHandler
import com.malaria.ml.TFLiteContext

/**
 * Application subclass — composition root for process-scoped wiring.
 *
 * Phase 3 wires the [TFLiteContext] application-context holder so the
 * shared-module `Classifier` actual can open `assets/models/Malaria_*.tflite`
 * without taking a [android.content.Context] argument in its constructor
 * (which would break iOS parity for the expect/actual signature).
 *
 * Spec §19 names this class as the composition root for application-scoped
 * state; per the no-ViewModels architecture (spec §4) services are still
 * injected via CompositionLocal from MainActivity, not from a DI container.
 */
class MalariaApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        TFLiteContext.install(this)
        ModelHashCache.attachAppContext(this)
    }

    override fun onTerminate() {
        try {
            NativeCrashHandler.uninstall()
        } catch (_: Throwable) {
            // Native library unavailable — nothing to uninstall.
        }
        super.onTerminate()
    }
}
