package com.malaria.android.data

import android.os.Build
import com.malaria.android.BuildConfig

/**
 * Small helper for stamping `appVersion` and `osVersion` on persisted
 * records. Both fields appear on [com.malaria.android.data.entities.Prediction]
 * and [com.malaria.android.data.entities.AuditEntry]. Mirrors
 * `iosApp/Persistence/BuildEnvironment.swift`.
 */
object BuildEnvironment {
    val appVersion: String get() = BuildConfig.VERSION_NAME ?: "unknown"
    val osVersion: String get() = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})"
}
