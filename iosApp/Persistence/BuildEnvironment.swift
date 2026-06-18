import Foundation

/// Small helper for stamping `appVersion` and `osVersion` on persisted
/// records. Both fields appear on `Prediction` and `AuditEntry`.
///
/// Reads via `ProcessInfo.processInfo`, NOT `UIDevice.current` — the
/// latter is `@MainActor`-isolated under Swift 6 strict concurrency and
/// this helper is consumed from non-isolated contexts (audit writes from
/// background queues, etc.). `ProcessInfo` is `nonisolated` and returns
/// the same OS version string.
enum BuildEnvironment {
    static var appVersion: String {
        (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "unknown"
    }

    static var osVersion: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "iOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }
}
