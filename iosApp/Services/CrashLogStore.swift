import Foundation
import Observation
@preconcurrency import Shared

/// Settings → Crash logs facade (spec §16).
///
/// Reads / lists the on-device JSON crash log files written by
/// `CrashLogWriter`. Sweeps logs older than 30 days on every init (spec
/// §16: "auto-expire after 30 days"). Audits each share as
/// `crash_log_shared` with the incident UUID.
@Observable
@MainActor
final class CrashLogStore {

    /// Spec §16: "auto-expire after 30 days (file modification time check
    /// on every app launch)".
    static let maxAgeSeconds: TimeInterval = 30 * 24 * 60 * 60

    private let directoryURL: URL
    private let auditLog: AuditLog
    private let authGate: AuthGate
    private let clinicians: ClinicianRepository

    /// Cached enumeration. Rebuilt on `refresh()` and after `didShare(…)`.
    private(set) var entries: [CrashLogEntry] = []

    init(
        directoryURL: URL? = nil,
        auditLog: AuditLog,
        authGate: AuthGate,
        clinicians: ClinicianRepository
    ) {
        // Default: `~/Documents/crashlogs/`.
        let fm = FileManager.default
        let documents = (try? fm.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        )) ?? URL(fileURLWithPath: NSTemporaryDirectory())
        self.directoryURL = directoryURL ?? documents.appendingPathComponent("crashlogs", isDirectory: true)
        self.auditLog = auditLog
        self.authGate = authGate
        self.clinicians = clinicians
        try? fm.createDirectory(at: self.directoryURL, withIntermediateDirectories: true)
        Self.sweepExpired(in: self.directoryURL)
        refresh()
    }

    /// Re-enumerate the directory. Newest first by mtime.
    func refresh() {
        entries = Self.enumerate(at: directoryURL)
    }

    /// File URL passed to `UIActivityViewController`. Returning the
    /// in-Documents URL directly works because the file already lives in
    /// the app sandbox and the share sheet just hands it off.
    func shareableURL(_ entry: CrashLogEntry) -> URL { entry.url }

    /// Decode a crash log on demand. Returns a Swift representation; the
    /// Settings detail view only renders timestamp + incident UUID
    /// per spec §16 ("Settings → Crash logs shows a list with timestamp
    /// and incident UUID").
    func read(_ entry: CrashLogEntry) throws -> DecodedCrashLog {
        let data = try Data(contentsOf: entry.url)
        return try JSONDecoder().decode(DecodedCrashLog.self, from: data)
    }

    /// Spec §16: "Sharing is audited as `CRASH_LOG_SHARED` with the
    /// incident UUID".
    func didShare(_ entry: CrashLogEntry) {
        let profile = (try? clinicians.current())
        auditLog.write(
            .crashLogShared,
            actorId: profile?.actorId ?? "unknown",
            actorRoleAtTime: profile?.role ?? "unknown",
            resourceType: "crash_log",
            resourceId: entry.incidentId,
            metadata: [:]
        )
    }

    /// Convenience for the section header readout.
    func count() -> Int { entries.count }

    // MARK: - Static helpers

    static func sweepExpired(in directory: URL) {
        let fm = FileManager.default
        let cutoff = Date().addingTimeInterval(-maxAgeSeconds)
        guard let contents = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { return }
        for url in contents where url.pathExtension == "json" {
            if let mtime = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate,
               mtime < cutoff {
                try? fm.removeItem(at: url)
            }
        }
    }

    private static func enumerate(at directory: URL) -> [CrashLogEntry] {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }
        let entries = contents
            .filter { $0.pathExtension == "json" }
            .compactMap { url -> CrashLogEntry? in
                let values = (try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey]))
                let mtime = values?.contentModificationDate ?? Date.distantPast
                let size = values?.fileSize ?? 0
                let incidentId = url.deletingPathExtension().lastPathComponent
                return CrashLogEntry(
                    incidentId: incidentId,
                    timestamp: mtime,
                    sizeBytes: size,
                    url: url
                )
            }
        return entries.sorted { $0.timestamp > $1.timestamp }
    }
}

/// Lightweight row model for the Crash logs list. Doesn't include the full
/// decoded body — that's lazy via `CrashLogStore.read(_:)`.
struct CrashLogEntry: Identifiable, Hashable, Sendable {
    let incidentId: String
    let timestamp: Date
    let sizeBytes: Int
    let url: URL

    var id: String { incidentId }
}

/// Decoded JSON shape mirroring the shared `CrashLogRecord`. Kept as a
/// Swift `Codable` for ergonomic decoding off the K/N type.
struct DecodedCrashLog: Codable, Sendable {
    let incidentId: String
    let timestampIso8601: String
    let appVersion: String
    let osVersion: String
    let deviceModelClass: String
    let stackTrace: String
    let recentActionTypes: [String]
    let memoryPressure: String
    let deviceUnlocked: Bool
}
