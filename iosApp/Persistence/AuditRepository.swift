import Foundation
import SwiftData

/// Append-only writer for `AuditEntry`. The public surface intentionally
/// exposes no update or delete methods — append-only by application
/// convention is spec §8's stated guarantee on both platforms.
///
/// `seq` is assigned at write time as `max(seq) + 1`. This is correct on
/// a single-threaded `@MainActor` ModelContext (the production setup);
/// concurrent writers would need stronger sequencing.
@MainActor
struct AuditRepository {
    let context: ModelContext

    /// Write a new audit entry. Returns the persisted entity (useful for
    /// tests that want to assert on the assigned `seq`).
    @discardableResult
    func write(
        action: AuditAction,
        actorId: String,
        actorRoleAtTime: String,
        resourceType: String? = nil,
        resourceId: String? = nil,
        metadata: [String: String] = [:],
        overrideContext: String? = nil,
        overrideReason: String? = nil,
        overrideNotes: String? = nil,
        contextReviewed: Bool? = nil,
        overrideActorInitials: String? = nil,
        timestamp: Date = Date()
    ) throws -> AuditEntry {
        let entry = AuditEntry(
            seq: try nextSeq(),
            timestamp: timestamp,
            actorId: actorId,
            actorRoleAtTime: actorRoleAtTime,
            action: action.canonical,
            resourceType: resourceType,
            resourceId: resourceId,
            metadataJson: encode(metadata),
            overrideContext: overrideContext,
            overrideReason: overrideReason,
            overrideNotes: overrideNotes,
            contextReviewed: contextReviewed,
            overrideActorInitials: overrideActorInitials,
            appVersion: BuildEnvironment.appVersion,
            osVersion: BuildEnvironment.osVersion
        )
        context.insert(entry)
        try context.save()
        return entry
    }

    /// Most recent N entries, newest first. The audit log viewer in
    /// History uses this with paging (spec §11).
    func recent(limit: Int = 200) throws -> [AuditEntry] {
        var descriptor = FetchDescriptor<AuditEntry>(
            sortBy: [SortDescriptor(\.seq, order: .reverse)]
        )
        descriptor.fetchLimit = limit
        return try context.fetch(descriptor)
    }

    /// Entries for a specific action — used in compliance review and
    /// also by `OnboardingState` to detect prior provisioning.
    func entries(forAction action: AuditAction) throws -> [AuditEntry] {
        let canonical = action.canonical
        let descriptor = FetchDescriptor<AuditEntry>(
            predicate: #Predicate { $0.action == canonical },
            sortBy: [SortDescriptor(\.seq, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    /// Total entry count — exposed for the export summary block.
    func count() throws -> Int {
        try context.fetchCount(FetchDescriptor<AuditEntry>())
    }

    // MARK: - Private

    private func nextSeq() throws -> Int {
        var descriptor = FetchDescriptor<AuditEntry>(
            sortBy: [SortDescriptor(\.seq, order: .reverse)]
        )
        descriptor.fetchLimit = 1
        let lastSeq = try context.fetch(descriptor).first?.seq ?? 0
        return lastSeq + 1
    }

    private func encode(_ metadata: [String: String]) -> String {
        guard !metadata.isEmpty,
              let data = try? JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    }
}
