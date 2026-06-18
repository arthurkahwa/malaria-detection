package com.malaria.permissions

import com.malaria.domain.ClinicianRole

/**
 * Role-based authorization for sensitive actions.
 *
 * The matrix is the canonical source of truth for both platforms.
 */
object Permissions {
    enum class Action {
        CHANGE_THRESHOLD,
        CHANGE_JURISDICTION,
        CHANGE_DEFAULT_MODEL,
        CHANGE_AUTO_LOGOUT,
        RESET_DEVICE,
        TRANSFER_ROLE,
        EXPORT_ALL_DATA,
        VIEW_AUDIT_LOG,
        CREATE_PREDICTION,
        OVERRIDE_PREDICTION,
        MARK_AS_DUPLICATE,
        RELABEL_SESSION
    }

    fun canPerform(role: ClinicianRole, action: Action): Boolean = when (action) {
        Action.CHANGE_THRESHOLD,
        Action.CHANGE_JURISDICTION,
        Action.CHANGE_DEFAULT_MODEL,
        Action.CHANGE_AUTO_LOGOUT,
        Action.RESET_DEVICE,
        Action.TRANSFER_ROLE -> role == ClinicianRole.ADMIN

        Action.EXPORT_ALL_DATA,
        Action.VIEW_AUDIT_LOG,
        Action.CREATE_PREDICTION,
        Action.OVERRIDE_PREDICTION,
        Action.MARK_AS_DUPLICATE,
        Action.RELABEL_SESSION -> role == ClinicianRole.ADMIN || role == ClinicianRole.MICROSCOPIST
    }
}
