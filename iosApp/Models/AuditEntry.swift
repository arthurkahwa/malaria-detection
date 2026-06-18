import Foundation
import SwiftData

/// Append-only audit event. The repository layer must never expose update
/// or delete operations on this type (spec §8). `seq` is a monotonic
/// per-device sequence assigned by `AuditRepository.write` at insert time.
@Model
final class AuditEntry {
    @Attribute(.unique) var id: String
    var seq: Int
    var timestamp: Date
    var actorId: String
    var actorRoleAtTime: String
    var action: String
    var resourceType: String?
    var resourceId: String?
    var metadataJson: String
    var overrideContext: String?
    var overrideReason: String?
    var overrideNotes: String?
    var contextReviewed: Bool?
    var overrideActorInitials: String?
    var appVersion: String
    var osVersion: String

    init(
        id: String = UUID().uuidString,
        seq: Int,
        timestamp: Date,
        actorId: String,
        actorRoleAtTime: String,
        action: String,
        resourceType: String? = nil,
        resourceId: String? = nil,
        metadataJson: String = "{}",
        overrideContext: String? = nil,
        overrideReason: String? = nil,
        overrideNotes: String? = nil,
        contextReviewed: Bool? = nil,
        overrideActorInitials: String? = nil,
        appVersion: String,
        osVersion: String
    ) {
        self.id = id
        self.seq = seq
        self.timestamp = timestamp
        self.actorId = actorId
        self.actorRoleAtTime = actorRoleAtTime
        self.action = action
        self.resourceType = resourceType
        self.resourceId = resourceId
        self.metadataJson = metadataJson
        self.overrideContext = overrideContext
        self.overrideReason = overrideReason
        self.overrideNotes = overrideNotes
        self.contextReviewed = contextReviewed
        self.overrideActorInitials = overrideActorInitials
        self.appVersion = appVersion
        self.osVersion = osVersion
    }
}
