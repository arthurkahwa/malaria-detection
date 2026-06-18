import Foundation
import SwiftData

/// Persists onboarding acknowledgements (spec §10). Each record pairs
/// with an `AuditEntry` recording the same event — the audit log is the
/// chain-of-custody, the consent record is the per-clinician acknowledgement
/// that supports a later "did this user accept document version X" query.
@MainActor
struct ConsentRepository {
    let context: ModelContext

    @discardableResult
    func record(
        actorId: String,
        consentType: ConsentType,
        documentVersion: String,
        value: String,
        timestamp: Date = Date()
    ) throws -> ConsentRecord {
        let record = ConsentRecord(
            timestamp: timestamp,
            actorId: actorId,
            consentType: consentType.canonical,
            documentVersion: documentVersion,
            value: value,
            appVersion: BuildEnvironment.appVersion
        )
        context.insert(record)
        try context.save()
        return record
    }

    /// All consent records for a clinician, newest-first.
    func records(for actorId: String) throws -> [ConsentRecord] {
        let descriptor = FetchDescriptor<ConsentRecord>(
            predicate: #Predicate { $0.actorId == actorId },
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    /// True iff the given clinician has acknowledged this consent type at
    /// the specified document version. Used during onboarding flow gating.
    func hasAccepted(_ type: ConsentType, version: String, actorId: String) throws -> Bool {
        let canonical = type.canonical
        let descriptor = FetchDescriptor<ConsentRecord>(
            predicate: #Predicate {
                $0.actorId == actorId
                    && $0.consentType == canonical
                    && $0.documentVersion == version
            }
        )
        return try context.fetchCount(descriptor) > 0
    }
}

/// Canonical consent types stored on `ConsentRecord.consentType`. Mirrors
/// spec §10 onboarding steps.
enum ConsentType: String, CaseIterable, Sendable {
    case hippocraticLicense = "hippocratic_license"
    case medicalDisclaimer = "medical_disclaimer"
    case lawfulBasis = "lawful_basis"
    case jurisdiction = "jurisdiction"

    var canonical: String { rawValue }
}
