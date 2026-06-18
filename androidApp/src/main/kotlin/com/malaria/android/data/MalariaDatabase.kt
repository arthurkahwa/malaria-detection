package com.malaria.android.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.malaria.android.data.dao.AuditDao
import com.malaria.android.data.dao.ClinicianDao
import com.malaria.android.data.dao.ConsentDao
import com.malaria.android.data.dao.PredictionDao
import com.malaria.android.data.entities.AuditEntry
import com.malaria.android.data.entities.ClinicianProfile
import com.malaria.android.data.entities.ConsentRecord
import com.malaria.android.data.entities.Prediction
import net.zetetic.database.sqlcipher.SupportOpenHelperFactory

/**
 * Room database for the malaria detector (spec §8, §19).
 *
 * Persisted under SQLCipher AES-256 with a passphrase derived in
 * [SecureKeyStore] using an Android Keystore-resident AES-GCM wrapping key
 * (StrongBox-preferred). The schema is exported to `androidApp/schemas` so
 * the JVM schema-drift test can diff it against `docs/SCHEMA.md`.
 *
 * @see net.zetetic.database.sqlcipher.SupportOpenHelperFactory the
 *      SQLCipher 4.6.1 integration that Room calls into for every open.
 */
@Database(
    entities = [
        Prediction::class,
        AuditEntry::class,
        ClinicianProfile::class,
        ConsentRecord::class,
    ],
    version = 1,
    exportSchema = true,
)
@TypeConverters(InstantConverter::class)
abstract class MalariaDatabase : RoomDatabase() {

    abstract fun predictionDao(): PredictionDao
    abstract fun auditDao(): AuditDao
    abstract fun clinicianDao(): ClinicianDao
    abstract fun consentDao(): ConsentDao

    companion object {
        private const val DATABASE_NAME = "malaria.db"

        /**
         * Build the production database. SQLCipher's native library is loaded
         * on first call (idempotent). The passphrase is fetched/created in
         * [SecureKeyStore] and zeroed by SQLCipher after the key is set on the
         * connection.
         */
        fun create(context: Context): MalariaDatabase {
            loadSqlCipherNative()
            val passphrase = SecureKeyStore.getOrCreateDatabaseKey(context)
            val factory = SupportOpenHelperFactory(passphrase)
            return Room.databaseBuilder(
                context.applicationContext,
                MalariaDatabase::class.java,
                DATABASE_NAME,
            )
                .openHelperFactory(factory)
                .build()
        }

        @Volatile
        private var nativeLoaded: Boolean = false

        private fun loadSqlCipherNative() {
            if (nativeLoaded) return
            synchronized(this) {
                if (nativeLoaded) return
                System.loadLibrary("sqlcipher")
                nativeLoaded = true
            }
        }
    }
}
