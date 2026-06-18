package com.malaria

import com.malaria.domain.Jurisdiction
import com.malaria.retention.RetentionPolicy
import kotlin.test.Test
import kotlin.test.assertEquals

class RetentionPolicyTest {

    @Test
    fun usHipaaIsSixYears() {
        assertEquals(6, RetentionPolicy.minimumYears(Jurisdiction.US_HIPAA))
    }

    @Test
    fun euGdprGermanyIsTenYears() {
        assertEquals(10, RetentionPolicy.minimumYears(Jurisdiction.EU_GDPR_DE))
    }

    @Test
    fun euGdprFranceIsTwentyYears() {
        assertEquals(20, RetentionPolicy.minimumYears(Jurisdiction.EU_GDPR_FR))
    }

    @Test
    fun euGdprGenericIsTenYears() {
        assertEquals(10, RetentionPolicy.minimumYears(Jurisdiction.EU_GDPR_GENERIC))
    }

    @Test
    fun kenyaDpaIsSevenYears() {
        assertEquals(7, RetentionPolicy.minimumYears(Jurisdiction.KE_DPA))
    }

    @Test
    fun otherIsSixYears() {
        assertEquals(6, RetentionPolicy.minimumYears(Jurisdiction.OTHER))
    }

    @Test
    fun everyJurisdictionHasAnEntry() {
        // Defensive: if a new jurisdiction is added, this test reminds us to
        // add a retention rule for it (RetentionPolicy is a `when` over the
        // enum, so a missing branch would already fail compilation, but this
        // also asserts the value is positive).
        for (j in Jurisdiction.entries) {
            val years = RetentionPolicy.minimumYears(j)
            assertEquals(true, years > 0, "Retention for $j must be positive (got $years)")
        }
    }
}
