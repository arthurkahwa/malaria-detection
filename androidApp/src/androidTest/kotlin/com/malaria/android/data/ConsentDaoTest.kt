// Requires API 36 Android emulator. Run via: ./gradlew :androidApp:connectedDebugAndroidTest
package com.malaria.android.data

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.malaria.android.data.dao.ConsentDao
import com.malaria.android.data.entities.ConsentType
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ConsentDaoTest {

    private lateinit var db: MalariaDatabase
    private lateinit var dao: ConsentDao

    @Before
    fun setUp() {
        db = TestSupport.inMemoryDatabase(TestSupport.context)
        dao = db.consentDao()
    }

    @After
    fun tearDown() { db.close() }

    @Test
    fun record_persists() = runTest {
        dao.record(
            actorId = "actor-1",
            consentType = ConsentType.HippocraticLicense,
            documentVersion = "v1",
            value = "accepted",
            appVersion = "0.1.0",
        )
        val all = dao.records("actor-1")
        assertEquals(1, all.size)
        assertEquals("hippocratic_license", all.first().consentType)
        assertEquals("v1", all.first().documentVersion)
    }

    @Test
    fun records_filtersByActor() = runTest {
        dao.record("actor-1", ConsentType.MedicalDisclaimer, "v1", "accepted", "0.1.0")
        dao.record("actor-2", ConsentType.MedicalDisclaimer, "v1", "accepted", "0.1.0")
        assertEquals(1, dao.records("actor-1").size)
    }

    @Test
    fun hasAccepted_returnsTrueAfterRecord() = runTest {
        assertFalse(dao.hasAccepted(ConsentType.LawfulBasis, "v1", "actor-1"))
        dao.record(
            actorId = "actor-1",
            consentType = ConsentType.LawfulBasis,
            documentVersion = "v1",
            value = "explicit_consent",
            appVersion = "0.1.0",
        )
        assertTrue(dao.hasAccepted(ConsentType.LawfulBasis, "v1", "actor-1"))
        // Different version → not accepted.
        assertFalse(dao.hasAccepted(ConsentType.LawfulBasis, "v2", "actor-1"))
    }
}
