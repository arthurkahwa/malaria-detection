import Foundation
import Observation

/// `@Observable` settings facade backed by the audit log (spec §11, §8).
///
/// Clinic-level configuration (name, jurisdiction, lawful basis) and the
/// inference policy (threshold, default model, auto-logout) live in the
/// audit log as the source of truth — they are written during Phase 1
/// admin provisioning (`clinic_configured`, `inference_policy_set`) and
/// updated by Settings edits via `threshold_changed`, `default_model_changed`,
/// `auto_logout_changed`. Holding the latest values here avoids re-parsing
/// metadata JSON on every render.
///
/// `language` lives in `UserDefaults` (`LanguagePreference`) because it
/// must survive `Reset device` (spec §10) where the audit log gets a fresh
/// `device_reprovisioned` entry but keeps every prior provisioning entry —
/// the chain-of-custody requirement.
@Observable
@MainActor
final class SettingsStore {

    // Clinic identity — read from the most recent `clinic_configured`
    // audit entry's metadata. Read-only in Settings per spec §11 (the
    // clinic config can only change via a full device reset).
    private(set) var clinicName: String?
    private(set) var jurisdiction: String?
    private(set) var lawfulBasis: String?

    // Inference policy — read from the most recent `inference_policy_set`
    // entry, then mutated through the Settings edit screens. Editable by
    // admin only per spec §11 + §9 (fresh biometric required).
    private(set) var threshold: Double = 0.3
    private(set) var defaultModelId: String = "BNLeaky_Keras"
    private(set) var autoLogoutMinutes: Int = 15

    /// Persisted UI language. Mirrors `LanguagePreference` raw value.
    private(set) var language: String = OnboardingLanguage.english.rawValue

    private let auditRepo: AuditRepository
    private let audit: AuditLog
    private let clinicians: ClinicianRepository

    init(auditRepo: AuditRepository, audit: AuditLog, clinicians: ClinicianRepository) {
        self.auditRepo = auditRepo
        self.audit = audit
        self.clinicians = clinicians
    }

    /// Hydrate from the audit log + `LanguagePreference`. Called once at
    /// composition root; safe to call again after a `Reset device`.
    func hydrate() {
        if let clinicEntry = try? auditRepo.entries(forAction: .clinicConfigured).first {
            let metadata = decode(clinicEntry.metadataJson)
            clinicName = metadata["clinic_name"]
            jurisdiction = metadata["jurisdiction"]
            lawfulBasis = metadata["lawful_basis"]
        } else {
            clinicName = nil
            jurisdiction = nil
            lawfulBasis = nil
        }

        if let policy = try? auditRepo.entries(forAction: .inferencePolicySet).first {
            let metadata = decode(policy.metadataJson)
            if let t = metadata["threshold"].flatMap(Double.init) { threshold = t }
            if let m = metadata["default_model"] { defaultModelId = m }
            if let a = metadata["auto_logout_minutes"].flatMap(Int.init) { autoLogoutMinutes = a }
        }

        // Walk audit log for the most-recent change-of-value entries so a
        // post-onboarding edit wins over the original `inference_policy_set`.
        if let last = try? auditRepo.entries(forAction: .thresholdChanged).first,
           let newValue = decode(last.metadataJson)["new_value"].flatMap(Double.init) {
            threshold = newValue
        }
        if let last = try? auditRepo.entries(forAction: .defaultModelChanged).first,
           let newValue = decode(last.metadataJson)["new_value"] {
            defaultModelId = newValue
        }
        if let last = try? auditRepo.entries(forAction: .autoLogoutChanged).first,
           let newValue = decode(last.metadataJson)["new_value"].flatMap(Int.init) {
            autoLogoutMinutes = newValue
        }

        language = LanguagePreference.get().rawValue
    }

    // MARK: - Edits (each writes the matching audit entry)

    /// Update the decision threshold. Writes a `threshold_changed` audit
    /// entry carrying old + new values per spec §11.
    func updateThreshold(_ newValue: Double) {
        let old = threshold
        threshold = newValue
        let actor = currentActor()
        audit.write(
            .thresholdChanged,
            actorId: actor.id,
            actorRoleAtTime: actor.role,
            metadata: ["old_value": String(old), "new_value": String(newValue)]
        )
    }

    func updateDefaultModel(_ newValue: String) {
        let old = defaultModelId
        defaultModelId = newValue
        let actor = currentActor()
        audit.write(
            .defaultModelChanged,
            actorId: actor.id,
            actorRoleAtTime: actor.role,
            metadata: ["old_value": old, "new_value": newValue]
        )
    }

    func updateAutoLogoutMinutes(_ newValue: Int) {
        let old = autoLogoutMinutes
        autoLogoutMinutes = newValue
        let actor = currentActor()
        audit.write(
            .autoLogoutChanged,
            actorId: actor.id,
            actorRoleAtTime: actor.role,
            metadata: ["old_value": String(old), "new_value": String(newValue)]
        )
    }

    func updateLanguage(_ newValue: OnboardingLanguage) {
        let old = language
        language = newValue.rawValue
        LanguagePreference.set(newValue)
        let actor = currentActor()
        audit.write(
            .languageChanged,
            actorId: actor.id,
            actorRoleAtTime: actor.role,
            metadata: ["old_value": old, "new_value": newValue.rawValue]
        )
    }

    /// Update microscopist initials. Writes a `profile_updated` audit entry
    /// with `field=initials` so a downstream auditor can scope queries.
    func updateInitials(_ newValue: String?) throws {
        guard let profile = try clinicians.current() else { return }
        try clinicians.updateInitials(newValue, on: profile)
        audit.write(
            .profileUpdated,
            actorId: profile.actorId,
            actorRoleAtTime: profile.role,
            resourceType: "clinician",
            resourceId: profile.actorId,
            metadata: ["field": "initials"]
        )
    }

    // MARK: - Private

    private struct Actor { let id: String; let role: String }

    private func currentActor() -> Actor {
        let profile = try? clinicians.current()
        return Actor(id: profile?.actorId ?? "unknown", role: profile?.role ?? "unknown")
    }

    private func decode(_ json: String) -> [String: String] {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        var out: [String: String] = [:]
        for (k, v) in obj {
            if let s = v as? String { out[k] = s }
            else { out[k] = "\(v)" }
        }
        return out
    }
}
