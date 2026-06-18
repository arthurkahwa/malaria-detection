package com.malaria

import com.malaria.domain.ClinicianRole
import com.malaria.permissions.Permissions
import com.malaria.permissions.Permissions.Action
import kotlin.test.Test
import kotlin.test.assertEquals

class PermissionsTest {

    /**
     * Expected matrix copied from the spec §5. Updates to the matrix must be
     * mirrored here.
     */
    private val expected: Map<Action, Set<ClinicianRole>> = mapOf(
        Action.CHANGE_THRESHOLD to setOf(ClinicianRole.ADMIN),
        Action.CHANGE_JURISDICTION to setOf(ClinicianRole.ADMIN),
        Action.CHANGE_DEFAULT_MODEL to setOf(ClinicianRole.ADMIN),
        Action.CHANGE_AUTO_LOGOUT to setOf(ClinicianRole.ADMIN),
        Action.RESET_DEVICE to setOf(ClinicianRole.ADMIN),
        Action.TRANSFER_ROLE to setOf(ClinicianRole.ADMIN),
        Action.EXPORT_ALL_DATA to setOf(ClinicianRole.ADMIN, ClinicianRole.MICROSCOPIST),
        Action.VIEW_AUDIT_LOG to setOf(ClinicianRole.ADMIN, ClinicianRole.MICROSCOPIST),
        Action.CREATE_PREDICTION to setOf(ClinicianRole.ADMIN, ClinicianRole.MICROSCOPIST),
        Action.OVERRIDE_PREDICTION to setOf(ClinicianRole.ADMIN, ClinicianRole.MICROSCOPIST),
        Action.MARK_AS_DUPLICATE to setOf(ClinicianRole.ADMIN, ClinicianRole.MICROSCOPIST),
        Action.RELABEL_SESSION to setOf(ClinicianRole.ADMIN, ClinicianRole.MICROSCOPIST)
    )

    @Test
    fun matrixCoversEveryActionAndRole() {
        for (action in Action.entries) {
            val allowed = expected.getValue(action)
            for (role in ClinicianRole.entries) {
                val actual = Permissions.canPerform(role, action)
                assertEquals(
                    expected = role in allowed,
                    actual = actual,
                    message = "canPerform($role, $action) expected=${role in allowed} but was $actual"
                )
            }
        }
    }

    @Test
    fun observerIsNeverPermitted() {
        for (action in Action.entries) {
            assertEquals(
                expected = false,
                actual = Permissions.canPerform(ClinicianRole.OBSERVER, action),
                message = "Observer must not be allowed to $action"
            )
        }
    }

    @Test
    fun adminIsAlwaysPermitted() {
        for (action in Action.entries) {
            assertEquals(
                expected = true,
                actual = Permissions.canPerform(ClinicianRole.ADMIN, action),
                message = "Admin must be allowed to $action"
            )
        }
    }
}
