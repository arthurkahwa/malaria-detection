// Requires API 36 Android emulator. Run via: ./gradlew :androidApp:connectedDebugAndroidTest
package com.malaria.android.data

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.malaria.android.data.dao.ClinicianDao
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ClinicianDaoTest {

    private lateinit var db: MalariaDatabase
    private lateinit var dao: ClinicianDao

    @Before
    fun setUp() {
        db = TestSupport.inMemoryDatabase(TestSupport.context)
        dao = db.clinicianDao()
    }

    @After
    fun tearDown() { db.close() }

    @Test
    fun current_isNullBeforeEnrollment() = runTest {
        assertNull(dao.current())
    }

    @Test
    fun enroll_createsActiveRow() = runTest {
        val profile = dao.enroll(role = "admin", initials = "AK")
        val current = dao.current()
        assertNotNull(current)
        assertEquals(profile.id, current!!.id)
        assertEquals("admin", current.role)
        assertEquals("AK", current.initials)
        assertFalse(current.biometricEnrolled)
    }

    @Test
    fun markBiometricEnrolled_flipsFlag() = runTest {
        val profile = dao.enroll(role = "admin")
        dao.markBiometricEnrolled(profile.id)
        assertTrue(dao.current()!!.biometricEnrolled)
    }

    @Test
    fun updateInitials_persists() = runTest {
        val profile = dao.enroll(role = "admin")
        dao.updateInitials(profile.id, "JM")
        assertEquals("JM", dao.current()!!.initials)
    }

    @Test
    fun wipe_removesAllProfiles() = runTest {
        dao.enroll(role = "admin")
        dao.wipe()
        assertNull(dao.current())
    }
}
