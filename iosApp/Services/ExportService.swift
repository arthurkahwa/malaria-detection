import Foundation
import Observation
import UIKit
@preconcurrency import Shared

/// Failures raised by [ExportService]. All paths emit an `export_failed`
/// audit entry before propagating.
enum ExportError: LocalizedError {
    case clinicianMissing
    case clinicConfigMissing
    case zipFailed(reason: String)
    case writeFailed(reason: String)

    var errorDescription: String? {
        switch self {
        case .clinicianMissing:
            "No clinician profile found. Complete onboarding before exporting."
        case .clinicConfigMissing:
            "Clinic configuration not found in the audit log. Complete admin provisioning first."
        case .zipFailed(let reason):
            "Could not create the export archive: \(reason)"
        case .writeFailed(let reason):
            "Could not write the export bundle: \(reason)"
        }
    }
}

/// Phase 13 export-bundle service (spec §14). The clinical-correctness logic
/// (JSON canonicalisation + HMAC signing) lives in the shared Kotlin module;
/// this `@Observable @MainActor` class is the iOS-side adapter that:
///
///   1. Fetches typed inputs from the four repositories + `SettingsStore`.
///   2. Converts SwiftData entities to the shared `Exported*` DTOs
///      (timestamps go through `ISO8601DateFormatter` with the
///      `.withInternetDateTime` options so the format matches the Android
///      `Instant.toString()` form byte-for-byte).
///   3. Calls `ExportBundleBuilder.build(...)` to produce the canonical
///      signed JSON.
///   4. Packs the JSON + a short `README.txt` into a ZIP via a tiny stored-
///      mode writer (`MinimalZipWriter`) and returns the file URL in a temp
///      directory.
///   5. Emits the `export_initiated` / `export_completed` / `export_failed`
///      audit chain per spec §14.
@Observable
@MainActor
final class ExportService {

    private let predictions: PredictionRepository
    private let audits: AuditRepository
    private let clinicians: ClinicianRepository
    private let consents: ConsentRepository
    private let settings: SettingsStore
    private let auditLog: AuditLog

    /// `UserDefaults` key for the cached device UUID fallback (spec §14
    /// wants a stable device identifier; `identifierForVendor` resets on
    /// app uninstall + reinstall which matches device-reset semantics, but
    /// in unit tests there is no UIDevice attached so we cache the value).
    private let deviceUuidDefaultsKey = "com.malaria.detector.exportDeviceUuid"

    init(
        predictions: PredictionRepository,
        audits: AuditRepository,
        clinicians: ClinicianRepository,
        consents: ConsentRepository,
        settings: SettingsStore,
        auditLog: AuditLog
    ) {
        self.predictions = predictions
        self.audits = audits
        self.clinicians = clinicians
        self.consents = consents
        self.settings = settings
        self.auditLog = auditLog
    }

    /// Produces a signed ZIP bundle and returns its file URL. The caller
    /// presents the share sheet (`UIActivityViewController`) over the URL.
    ///
    /// Side effects:
    ///   - On entry: writes an `export_initiated` audit entry.
    ///   - On success: writes `export_completed` with `size` and `signature`
    ///     metadata.
    ///   - On any thrown error: writes `export_failed` with `reason`
    ///     metadata.
    func generateBundle() async throws -> URL {
        let actor = try? clinicians.current()
        let actorId = actor?.actorId ?? "unknown"
        let actorRole = actor?.role ?? "unknown"

        auditLog.write(.exportInitiated, actorId: actorId, actorRoleAtTime: actorRole)

        do {
            let build = try await buildBundle(actorId: actorId)
            try MinimalZipWriter.writeArchive(
                entries: [
                    (name: "export.json", data: Data(build.signedJson.utf8)),
                    (name: "README.txt", data: Data(build.readme.utf8))
                ],
                to: build.bundleURL
            )
            let attrs = try? FileManager.default.attributesOfItem(atPath: build.bundleURL.path)
            let size = (attrs?[.size] as? NSNumber)?.intValue ?? 0
            auditLog.write(
                .exportCompleted,
                actorId: actorId,
                actorRoleAtTime: actorRole,
                metadata: ["size": String(size), "signature": build.signature]
            )
            return build.bundleURL
        } catch {
            auditLog.write(
                .exportFailed,
                actorId: actorId,
                actorRoleAtTime: actorRole,
                metadata: ["reason": error.localizedDescription]
            )
            throw error
        }
    }

    // MARK: - Internals

    /// Output of the JSON+naming step. The ZIP write is a separate step so
    /// the audit-completion metadata can record the file size and signature
    /// in one place.
    struct BundleBuild {
        let bundleURL: URL
        let signedJson: String
        let signature: String
        let readme: String
    }

    private func buildBundle(actorId: String) async throws -> BundleBuild {
        guard let clinicName = settings.clinicName,
              let jurisdiction = settings.jurisdiction,
              let lawfulBasis = settings.lawfulBasis else {
            throw ExportError.clinicConfigMissing
        }

        let exportDate = Date()
        let exportTimestamp = Self.iso8601(exportDate)

        // ----- Snapshot persistence into the shared DTO types -----------
        let predictionsRows = try predictions.recent(limit: Int.max)
        let auditRows = try audits.recent(limit: Int.max)
        let allProfiles = try [clinicians.current()].compactMap { $0 }
        let consentRecords = try consents.records(for: actorId)

        let dtoPredictions = predictionsRows.map(Self.toDTO(_:))
        let dtoAudits = auditRows.map(Self.toDTO(_:))
        let dtoProfiles = allProfiles.map(Self.toDTO(_:))

        let summary = ExportSummary(
            predictionCount: Int32(predictionsRows.count),
            sessionCount: Int32(Set(predictionsRows.map { $0.sessionId }).count),
            auditEntryCount: Int32(auditRows.count),
            consentRecordCount: Int32(consentRecords.count),
            firstPredictionAt: predictionsRows.map { Self.iso8601($0.timestamp) }.min(),
            lastPredictionAt: predictionsRows.map { Self.iso8601($0.timestamp) }.max()
        )

        // ----- Sign + serialize via the shared module -------------------
        let deviceUuid = Self.deviceUuid(defaultsKey: deviceUuidDefaultsKey)
        let signedJson = ExportBundleBuilder().build(
            exportTimestamp: exportTimestamp,
            exportedByActorId: actorId,
            deviceUuid: deviceUuid,
            platform: "ios",
            clinicName: clinicName,
            jurisdiction: jurisdiction,
            lawfulBasis: lawfulBasis,
            appVersion: BuildEnvironment.appVersion,
            osVersion: BuildEnvironment.osVersion,
            summary: summary,
            clinicianProfiles: dtoProfiles,
            predictions: dtoPredictions,
            auditLog: dtoAudits
        )

        let signature = Self.extractSignature(from: signedJson) ?? "unknown"

        let timestampSlug = Self.timestampSlug(exportDate)
        let prefix = String(deviceUuid.prefix(8))
        let filename = "malaria-detector-export-\(prefix)-\(timestampSlug).zip"
        let bundleURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(filename)
        try? FileManager.default.removeItem(at: bundleURL)

        let readme = Self.readmeText(
            exportTimestamp: exportTimestamp,
            clinicName: clinicName,
            summary: summary
        )

        return BundleBuild(
            bundleURL: bundleURL,
            signedJson: signedJson,
            signature: signature,
            readme: readme
        )
    }

    private static func extractSignature(from json: String) -> String? {
        let marker = "\"signature\":\""
        guard let start = json.range(of: marker, options: .backwards) else { return nil }
        let after = json[start.upperBound...]
        guard let end = after.firstIndex(of: "\"") else { return nil }
        return String(after[..<end])
    }

    // MARK: - DTO conversion

    private static func toDTO(_ p: Prediction) -> ExportedPrediction {
        ExportedPrediction(
            id: p.id,
            sessionId: p.sessionId,
            timestamp: iso8601(p.timestamp),
            modelId: p.modelId,
            modelVersion: p.modelVersion,
            parasitizedProb: p.parasitizedProb,
            uninfectedProb: p.uninfectedProb,
            label: p.label,
            threshold: p.threshold,
            flaggedForReview: p.flaggedForReview,
            inferenceMs: Int32(p.inferenceMs),
            imageHash: p.imageHash,
            clinicianOverride: p.clinicianOverride,
            overrideContext: p.overrideContext,
            duplicateOfId: p.duplicateOfId,
            sessionLabel: p.sessionLabel,
            appVersion: p.appVersion,
            osVersion: p.osVersion
        )
    }

    private static func toDTO(_ e: AuditEntry) -> ExportedAuditEntry {
        ExportedAuditEntry(
            id: e.id,
            seq: Int64(e.seq),
            timestamp: iso8601(e.timestamp),
            actorId: e.actorId,
            actorRoleAtTime: e.actorRoleAtTime,
            action: e.action,
            resourceType: e.resourceType,
            resourceId: e.resourceId,
            metadataJson: e.metadataJson,
            overrideContext: e.overrideContext,
            overrideReason: e.overrideReason,
            overrideNotes: e.overrideNotes,
            contextReviewed: e.contextReviewed.map { KotlinBoolean(value: $0) },
            overrideActorInitials: e.overrideActorInitials,
            appVersion: e.appVersion,
            osVersion: e.osVersion
        )
    }

    private static func toDTO(_ c: ClinicianProfile) -> ExportedClinicianProfile {
        ExportedClinicianProfile(
            actorId: c.actorId,
            role: c.role,
            initials: c.initials,
            enrolledAt: iso8601(c.enrolledAt),
            biometricEnrolled: c.biometricEnrolled
        )
    }

    // MARK: - Formatters

    private static let iso8601Formatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    private static func iso8601(_ date: Date) -> String {
        iso8601Formatter.string(from: date)
    }

    private static func timestampSlug(_ date: Date) -> String {
        // `2026-05-21T14-32-08Z` form — derived from the ISO-8601 with the
        // colons replaced (filesystem-safe).
        let raw = iso8601(date)
        return raw.replacingOccurrences(of: ":", with: "-")
    }

    static func deviceUuid(defaultsKey: String) -> String {
        if let vendor = UIDevice.current.identifierForVendor?.uuidString {
            return vendor
        }
        // Test or simulator fallback — cache a UUID in UserDefaults so the
        // export is reproducible across runs.
        let defaults = UserDefaults.standard
        if let cached = defaults.string(forKey: defaultsKey), !cached.isEmpty {
            return cached
        }
        let generated = UUID().uuidString
        defaults.set(generated, forKey: defaultsKey)
        return generated
    }

    private static func readmeText(
        exportTimestamp: String,
        clinicName: String,
        summary: ExportSummary
    ) -> String {
        """
        Malaria Detector export bundle
        ==============================
        Exported at:  \(exportTimestamp)
        Clinic:       \(clinicName)
        Predictions:  \(summary.predictionCount)
        Sessions:     \(summary.sessionCount)
        Audit rows:   \(summary.auditEntryCount)
        Consents:     \(summary.consentRecordCount)

        Contents:
          - export.json   — signed bundle (HMAC-SHA256, see spec §14)
          - README.txt    — this file

        Verifying the signature:
          1. Open export.json, strip the trailing `,"signature":"<hex>"}`.
          2. Re-serialise via the shared ExportBundleBuilder JSON config.
          3. Recompute the HMAC over the unsigned form using the device
             UUID + exportTimestamp salt.
          4. Compare to the original signature field.

        Generated by Malaria Detector v1. No images are included in this
        bundle — imageHash is the only durable trace of the analysed
        cells (spec §8).
        """
    }
}
