package com.malaria.android.data

import androidx.room.TypeConverter
import kotlinx.datetime.Instant

/**
 * Room type converter for [kotlinx.datetime.Instant] ↔ epoch milliseconds.
 *
 * The rest of the shared module uses `kotlinx.datetime.Instant` (see
 * `shared/src/commonMain/kotlin/com/malaria/session/SessionGrouping.kt`),
 * so the persistence layer follows suit rather than importing
 * `java.time.Instant`. Epoch ms is the lowest common denominator agreed
 * with iOS, where SwiftData stores `Date` as the same UTC ms.
 */
class InstantConverter {

    @TypeConverter
    fun fromEpochMs(epochMs: Long?): Instant? =
        epochMs?.let { Instant.fromEpochMilliseconds(it) }

    @TypeConverter
    fun toEpochMs(instant: Instant?): Long? =
        instant?.toEpochMilliseconds()
}
