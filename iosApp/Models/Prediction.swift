import Foundation
import SwiftData

/// One row per inference event. Persisted under `NSFileProtectionComplete`.
/// Schema mirrors `docs/SCHEMA.md`.
///
/// No image bytes ever land here — `imageHash` (SHA-256 of preprocessed
/// input) is the only durable trace of the analysed cell.
@Model
final class Prediction {
    @Attribute(.unique) var id: String
    var sessionId: String
    var timestamp: Date
    var modelId: String
    var modelVersion: String
    var parasitizedProb: Double
    var uninfectedProb: Double
    var label: String
    var threshold: Double
    var flaggedForReview: Bool
    var inferenceMs: Int
    var imageHash: String
    var clinicianOverride: String?
    var overrideContext: String?
    var duplicateOfId: String?
    var sessionLabel: String?
    var appVersion: String
    var osVersion: String

    init(
        id: String = UUID().uuidString,
        sessionId: String,
        timestamp: Date,
        modelId: String,
        modelVersion: String,
        parasitizedProb: Double,
        uninfectedProb: Double,
        label: String,
        threshold: Double,
        flaggedForReview: Bool,
        inferenceMs: Int,
        imageHash: String,
        clinicianOverride: String? = nil,
        overrideContext: String? = nil,
        duplicateOfId: String? = nil,
        sessionLabel: String? = nil,
        appVersion: String,
        osVersion: String
    ) {
        self.id = id
        self.sessionId = sessionId
        self.timestamp = timestamp
        self.modelId = modelId
        self.modelVersion = modelVersion
        self.parasitizedProb = parasitizedProb
        self.uninfectedProb = uninfectedProb
        self.label = label
        self.threshold = threshold
        self.flaggedForReview = flaggedForReview
        self.inferenceMs = inferenceMs
        self.imageHash = imageHash
        self.clinicianOverride = clinicianOverride
        self.overrideContext = overrideContext
        self.duplicateOfId = duplicateOfId
        self.sessionLabel = sessionLabel
        self.appVersion = appVersion
        self.osVersion = osVersion
    }
}
