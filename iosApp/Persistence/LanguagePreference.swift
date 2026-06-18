import Foundation

/// Persisted UI language for onboarding (spec §10 Phase 1 step 1).
///
/// Stored in `UserDefaults` — unencrypted because language choice is not
/// sensitive and must survive a `Reset device` operation (spec §10
/// re-onboarding). Phase 12 will add Swahili/French/Portuguese; for v0.1
/// only English is shipped in `Localizable.xcstrings`, but the enum is
/// already in place so the wizard step renders the full four-option
/// picker per spec.
enum OnboardingLanguage: String, CaseIterable, Sendable, Identifiable {
    case english = "en"
    case swahili = "sw"
    case french = "fr"
    case portuguese = "pt"

    var id: String { rawValue }

    /// English display label. Per spec §15 these display strings stay in
    /// English because translating "Swahili" into Swahili-as-rendered-in
    /// -the-app would make the picker unusable to a fresh device admin.
    var displayName: String {
        switch self {
        case .english: "English"
        case .swahili: "Swahili"
        case .french: "French"
        case .portuguese: "Portuguese"
        }
    }
}

/// Thin wrapper around `UserDefaults` so call sites don't repeat the key.
/// Stable across resets per spec §10.
enum LanguagePreference {

    /// `UserDefaults` key. Kept in one place so a future migration can
    /// rename it from a single edit.
    static let key = "onboarding.language"

    static func get(defaults: UserDefaults = .standard) -> OnboardingLanguage {
        guard let raw = defaults.string(forKey: key),
              let language = OnboardingLanguage(rawValue: raw)
        else {
            return .english
        }
        return language
    }

    static func set(_ language: OnboardingLanguage, defaults: UserDefaults = .standard) {
        defaults.set(language.rawValue, forKey: key)
    }
}
