import Foundation
import Observation
@preconcurrency import Shared

/// Thin `@Observable` wrapper around `AuditRepository`. Lives at the
/// composition root and is injected via `@Environment` so any service or
/// view can write an audit entry. The split lets a downstream deployer
/// add chain-hashing (spec §8 / §18) by replacing the repository call
/// site without touching every caller.
///
/// Side-effect (spec §16): every successful write also pushes the action's
/// canonical string into the shared `RecentActionRing` so a future crash
/// log captures the last 50 audit action types without having to touch
/// the database from the signal handler. The ring is in the shared module
/// to keep the contract identical across iOS and Android.
@Observable
@MainActor
final class AuditLog {
    private let repository: AuditRepository

    /// Optional `lastWriteError` lets views surface a write failure
    /// (rare, but the spec is explicit that persistence errors should
    /// not be silently swallowed).
    private(set) var lastWriteError: Error?

    init(repository: AuditRepository) {
        self.repository = repository
    }

    @discardableResult
    func write(
        _ action: AuditAction,
        actorId: String,
        actorRoleAtTime: String,
        resourceType: String? = nil,
        resourceId: String? = nil,
        metadata: [String: String] = [:],
        overrideContext: String? = nil,
        overrideReason: String? = nil,
        overrideNotes: String? = nil,
        contextReviewed: Bool? = nil,
        overrideActorInitials: String? = nil
    ) -> AuditEntry? {
        do {
            let entry = try repository.write(
                action: action,
                actorId: actorId,
                actorRoleAtTime: actorRoleAtTime,
                resourceType: resourceType,
                resourceId: resourceId,
                metadata: metadata,
                overrideContext: overrideContext,
                overrideReason: overrideReason,
                overrideNotes: overrideNotes,
                contextReviewed: contextReviewed,
                overrideActorInitials: overrideActorInitials
            )
            lastWriteError = nil
            // Spec §16: feed the canonical action string into the shared
            // ring so the crash handler can capture the last 50 actions.
            // The ring deliberately stores ONLY the action string — no
            // resource id, actor id, metadata, or timestamp — so privacy
            // invariants of the crash log hold by construction.
            RecentActionRing.companion.shared.push(action: action.canonical)
            return entry
        } catch {
            lastWriteError = error
            return nil
        }
    }
}
