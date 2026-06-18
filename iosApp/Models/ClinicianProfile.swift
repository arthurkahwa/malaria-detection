import Foundation
import SwiftData

/// The device's single clinician (v1 is single-clinician per spec §9).
/// `actorId` is pseudonymous — never linked to PII inside the app. A clinic
/// admin keeps the UUID → person mapping outside the app's scope.
@Model
final class ClinicianProfile {
    @Attribute(.unique) var id: String
    var actorId: String
    var role: String
    var initials: String?
    var enrolledAt: Date
    var biometricEnrolled: Bool

    init(
        id: String = UUID().uuidString,
        actorId: String,
        role: String,
        initials: String? = nil,
        enrolledAt: Date,
        biometricEnrolled: Bool = false
    ) {
        self.id = id
        self.actorId = actorId
        self.role = role
        self.initials = initials
        self.enrolledAt = enrolledAt
        self.biometricEnrolled = biometricEnrolled
    }
}
