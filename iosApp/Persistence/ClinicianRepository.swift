import Foundation
import SwiftData

/// CRUD for `ClinicianProfile`. v1 is single-clinician (spec §9) so the
/// repository exposes a `current()` shortcut that returns the sole row;
/// multi-clinician support is a v2 schema migration.
@MainActor
struct ClinicianRepository {
    let context: ModelContext

    /// The device's clinician, or nil if onboarding hasn't completed.
    func current() throws -> ClinicianProfile? {
        var descriptor = FetchDescriptor<ClinicianProfile>(
            sortBy: [SortDescriptor(\.enrolledAt, order: .forward)]
        )
        descriptor.fetchLimit = 1
        return try context.fetch(descriptor).first
    }

    @discardableResult
    func enroll(role: String, initials: String? = nil) throws -> ClinicianProfile {
        let actorId = UUID().uuidString
        let profile = ClinicianProfile(
            id: actorId,
            actorId: actorId,
            role: role,
            initials: initials,
            enrolledAt: Date(),
            biometricEnrolled: false
        )
        context.insert(profile)
        try context.save()
        return profile
    }

    func updateInitials(_ initials: String?, on profile: ClinicianProfile) throws {
        profile.initials = initials
        try context.save()
    }

    func markBiometricEnrolled(_ profile: ClinicianProfile) throws {
        profile.biometricEnrolled = true
        try context.save()
    }

    /// Wipe the clinician row. Used by `Settings → Reset device` per
    /// spec §10. Audit entries are preserved — only clinician-level
    /// state is removed.
    func wipe() throws {
        let descriptor = FetchDescriptor<ClinicianProfile>()
        let all = try context.fetch(descriptor)
        for p in all { context.delete(p) }
        try context.save()
    }
}
