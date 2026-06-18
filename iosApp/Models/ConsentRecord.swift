import Foundation
import SwiftData

/// Onboarding acknowledgements (Hippocratic License, medical-device
/// disclaimer, lawful-basis selection, jurisdiction selection). Each record
/// pairs with an `AuditEntry` recording the same event.
///
/// `documentVersion` locks the acknowledgement to specific shown wording so
/// a later legal-text revision doesn't retroactively claim prior consent.
@Model
final class ConsentRecord {
    @Attribute(.unique) var id: String
    var timestamp: Date
    var actorId: String
    var consentType: String
    var documentVersion: String
    var value: String
    var appVersion: String

    init(
        id: String = UUID().uuidString,
        timestamp: Date,
        actorId: String,
        consentType: String,
        documentVersion: String,
        value: String,
        appVersion: String
    ) {
        self.id = id
        self.timestamp = timestamp
        self.actorId = actorId
        self.consentType = consentType
        self.documentVersion = documentVersion
        self.value = value
        self.appVersion = appVersion
    }
}
