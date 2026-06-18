// Requires API 36 Android emulator. Run via: ./gradlew :androidApp:connectedDebugAndroidTest
package com.malaria.android.data

import android.content.Context
import androidx.room.Room
import androidx.test.platform.app.InstrumentationRegistry
import com.malaria.android.data.entities.Prediction
import kotlinx.datetime.Instant
import java.util.UUID

/**
 * In-memory [MalariaDatabase] for instrumented tests so each test is
 * isolated and fast. The schema-drift snapshot test (JVM-side) is the
 * official mitigation for any drift between this in-memory schema and
 * production, so test/prod parity here doesn't need to be re-asserted.
 *
 * Note: no SQLCipher passphrase. Encryption is verified separately by
 * [EncryptionVerificationTest] which uses a real on-disk database.
 */
object TestSupport {

    fun inMemoryDatabase(context: Context): MalariaDatabase =
        Room.inMemoryDatabaseBuilder(context, MalariaDatabase::class.java)
            .allowMainThreadQueries()
            .build()

    val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    fun samplePrediction(
        id: String = UUID.randomUUID().toString(),
        sessionId: String = UUID.randomUUID().toString(),
        timestampMs: Long = System.currentTimeMillis(),
        parasitizedProb: Double = 0.85,
        label: String = "Parasitized",
        threshold: Double = 0.3,
        flaggedForReview: Boolean = false,
        clinicianOverride: String? = null,
        duplicateOfId: String? = null,
    ): Prediction = Prediction(
        id = id,
        sessionId = sessionId,
        timestamp = Instant.fromEpochMilliseconds(timestampMs),
        modelId = "BNLeaky_Keras",
        modelVersion = "BNLeaky_Keras",
        parasitizedProb = parasitizedProb,
        uninfectedProb = 1.0 - parasitizedProb,
        label = label,
        threshold = threshold,
        flaggedForReview = flaggedForReview,
        inferenceMs = 42,
        imageHash = "a".repeat(64),
        clinicianOverride = clinicianOverride,
        duplicateOfId = duplicateOfId,
        appVersion = "0.1.0-test",
        osVersion = "Android Test",
    )
}
