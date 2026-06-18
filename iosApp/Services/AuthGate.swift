import Foundation
import LocalAuthentication
import Observation

/// Reason a session was locked. Recorded in the audit log per spec §9.
enum LockReason: Sendable {
    case background
    case manual
    case timeout

    fileprivate var auditAction: AuditAction {
        switch self {
        case .background: .sessionRelockedBackground
        case .manual: .sessionRelockedManual
        case .timeout: .sessionRelockedTimeout
        }
    }
}

/// Failures raised during biometric unlock.
enum AuthError: LocalizedError {
    case noBiometricsConfigured
    case userCancelled
    case authenticationFailed(reason: String)

    var errorDescription: String? {
        switch self {
        case .noBiometricsConfigured:
            "Set up Face ID, Touch ID, or a device passcode in Settings before using the app."
        case .userCancelled:
            "Authentication cancelled."
        case .authenticationFailed(let reason):
            "Authentication failed: \(reason)"
        }
    }
}

/// Biometric/passcode gate on top of `LAContext.deviceOwnerAuthentication`.
/// Per spec §9: Face ID, Touch ID, or device passcode unlock the app and
/// hold the session until backgrounded, manually locked, or auto-logged-out
/// (the SessionTimer wiring lands in Phase 6).
@Observable
@MainActor
final class AuthGate {

    enum State: Sendable {
        case locked
        case unlocked(sessionStart: Date)
        case provisionedUnclaimed
    }

    private(set) var state: State = .locked

    private let audit: AuditLog
    private let clinicians: ClinicianRepository

    init(audit: AuditLog, clinicians: ClinicianRepository) {
        self.audit = audit
        self.clinicians = clinicians
    }

    /// Refreshes initial state from persistence on app launch.
    func bootstrap() {
        do {
            let profile = try clinicians.current()
            if profile == nil || profile?.biometricEnrolled == false {
                state = .provisionedUnclaimed
            } else {
                state = .locked
            }
        } catch {
            state = .locked
        }
    }

    /// Present the system biometric prompt. Throws `AuthError` on failure.
    func unlock(reason: String = "Unlock Malaria Detector") async throws {
        let laContext = LAContext()
        var error: NSError?
        guard laContext.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error) else {
            try recordAuthFailure(reason: error?.localizedDescription ?? "unsupported")
            throw AuthError.noBiometricsConfigured
        }

        do {
            try await laContext.evaluatePolicy(.deviceOwnerAuthentication, localizedReason: reason)
        } catch let laError as LAError {
            if laError.code == .userCancel || laError.code == .systemCancel || laError.code == .appCancel {
                try recordAuthFailure(reason: "cancelled")
                throw AuthError.userCancelled
            }
            try recordAuthFailure(reason: laError.localizedDescription)
            throw AuthError.authenticationFailed(reason: laError.localizedDescription)
        }

        state = .unlocked(sessionStart: Date())
        let profile = try clinicians.current()
        audit.write(
            .sessionUnlocked,
            actorId: profile?.actorId ?? "unknown",
            actorRoleAtTime: profile?.role ?? "unknown"
        )
        audit.write(
            .authSuccess,
            actorId: profile?.actorId ?? "unknown",
            actorRoleAtTime: profile?.role ?? "unknown"
        )
    }

    func lockSession(reason: LockReason) {
        state = .locked
        let profile = try? clinicians.current()
        audit.write(
            reason.auditAction,
            actorId: profile?.actorId ?? "unknown",
            actorRoleAtTime: profile?.role ?? "unknown"
        )
    }

    /// Auto-logout via SessionTimer is intentionally disabled for the
    /// technology preview. The SessionTimer class exists in shared/ and
    /// the setting is configurable in Settings, but the foreground timer
    /// is not wired — only manual lock and scene-phase backgrounding fire.
    func touch() {
        // Auto-logout intentionally deactivated (technology preview).
        // Wire SessionTimer.touch() here when auto-logout is enabled.
    }

    private func recordAuthFailure(reason: String) throws {
        let profile = try clinicians.current()
        audit.write(
            .authFailure,
            actorId: profile?.actorId ?? "unknown",
            actorRoleAtTime: profile?.role ?? "unknown",
            metadata: ["reason": reason]
        )
    }
}
