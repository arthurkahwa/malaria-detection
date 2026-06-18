package com.malaria.retention

import com.malaria.domain.Jurisdiction

/**
 * Advisory retention policy. Used to inform the clinician of the minimum
 * recommended retention period for their jurisdiction. **Not** auto-enforced
 * in v1; deletion is deliberate, deployer-driven, audited.
 */
object RetentionPolicy {
    /** Minimum number of years records should be retained for [j]. */
    fun minimumYears(j: Jurisdiction): Int = when (j) {
        Jurisdiction.US_HIPAA -> 6
        Jurisdiction.EU_GDPR_DE -> 10
        Jurisdiction.EU_GDPR_FR -> 20
        Jurisdiction.EU_GDPR_GENERIC -> 10
        Jurisdiction.KE_DPA -> 7
        Jurisdiction.OTHER -> 6
    }
}
