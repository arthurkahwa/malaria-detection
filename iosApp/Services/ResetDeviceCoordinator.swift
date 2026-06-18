import Foundation
import Observation

/// Coordinates the spec §10 "Reset device" flow.
///
/// Holds non-`@Observable` collaborators (the SwiftData-backed repositories
/// are value-type structs that can't participate in SwiftUI's environment
/// type-based injection directly) so the Settings UI can request a wipe
/// without re-discovering them at the call site.
///
/// The actual biometric prompt is initiated by the calling view via
/// `AuthGate.unlock(reason:)` — this coordinator only owns the wipe +
/// audit + onboarding-reset sequence after the prompt resolves.
@Observable
@MainActor
final class ResetDeviceCoordinator {

    private let clinicians: ClinicianRepository
    private let audit: AuditLog
    private let onboarding: OnboardingState
    private let settings: SettingsStore

    init(
        clinicians: ClinicianRepository,
        audit: AuditLog,
        onboarding: OnboardingState,
        settings: SettingsStore
    ) {
        self.clinicians = clinicians
        self.audit = audit
        self.onboarding = onboarding
        self.settings = settings
    }

    /// Execute the wipe. Caller must have already triggered a fresh
    /// biometric prompt and double-confirmation per spec §10.
    ///
    /// Order is significant:
    ///   1. Capture the wiped actor id while the row still exists.
    ///   2. Wipe the clinician row (consents preserved for now —
    ///      see spec §10's "clinic-level config preserved" guarantee:
    ///      predictions + audit + clinic-level consents stay; only the
    ///      clinician identity is removed).
    ///   3. Write the `device_reprovisioned` audit entry (chain-of-
    ///      custody) BEFORE flipping `OnboardingState.phase` — so the
    ///      audit row records the actorId at the moment of wipe.
    ///   4. Reset `OnboardingState` and re-hydrate `SettingsStore`.
    func performReset() throws {
        let wipedActorId = (try? clinicians.current())?.actorId ?? "unknown"
        let wipedRole = (try? clinicians.current())?.role ?? "unknown"

        try clinicians.wipe()

        audit.write(
            .deviceReprovisioned,
            actorId: wipedActorId,
            actorRoleAtTime: wipedRole,
            resourceType: "clinician",
            resourceId: wipedActorId,
            metadata: ["wiped_actor_id": wipedActorId]
        )

        onboarding.reset()
        settings.hydrate()
    }
}
