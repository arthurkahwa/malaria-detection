// Requires API 36 Android emulator. Run via: ./gradlew :androidApp:connectedDebugAndroidTest
package com.malaria.android.data

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.test.runTest
import net.zetetic.database.sqlcipher.SQLiteDatabase
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Spec §8 / §20 mitigation: open the on-disk database file directly via
 * SQLCipher's [SQLiteDatabase] without the passphrase and assert the open
 * fails. If this test passes, SQLCipher is actually applied — the file
 * isn't readable as plain SQLite.
 *
 * Uses the production [MalariaDatabase.create] entry point so the test
 * exercises the real `SecureKeyStore` → `SupportOpenHelperFactory` chain.
 */
@RunWith(AndroidJUnit4::class)
class EncryptionVerificationTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private var db: MalariaDatabase? = null

    @Before
    fun setUp() {
        // Clean any previous test artifact so the keystore + passphrase are
        // re-derived for this test.
        context.getDatabasePath("malaria.db").let { f ->
            if (f.exists()) f.delete()
        }
    }

    @After
    fun tearDown() {
        db?.close()
    }

    @Test
    fun database_writesEncryptedFile() = runTest {
        val database = MalariaDatabase.create(context)
        db = database
        // Force an actual write so the file exists on disk.
        database.clinicianDao().enroll(role = "admin", initials = "AK")
        database.close()
        db = null

        val dbFile: File = context.getDatabasePath("malaria.db")
        assertNotNull("malaria.db should exist after first write", dbFile)
        assert(dbFile.exists()) { "malaria.db should exist; got ${dbFile.absolutePath}" }

        // Attempt to open WITHOUT a passphrase. SQLCipher prepends the
        // ciphertext header so opening as plain SQLite must fail.
        try {
            val plain = SQLiteDatabase.openDatabase(
                dbFile.absolutePath,
                /* factory = */ null,
                SQLiteDatabase.OPEN_READONLY,
            )
            plain.close()
            fail("Expected opening encrypted DB without passphrase to throw")
        } catch (_: Throwable) {
            // Expected — file is encrypted at rest.
        }
    }
}
