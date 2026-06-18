import Foundation
import Observation
@preconcurrency import Shared

/// Loads `model_registry.json` from the iOS bundle and parses it via
/// the shared `ModelRegistry`. Per spec §5 ("Model registry") and §7.
@Observable
@MainActor
final class ModelRegistryService {

    enum LoadError: Error, Sendable {
        case resourceMissing
        case decodingFailed(message: String)
    }

    private(set) var registry: ModelRegistry

    init() throws {
        guard let url = Bundle.main.url(
            forResource: "model_registry",
            withExtension: "json"
        ) else {
            throw LoadError.resourceMissing
        }
        let data = try Data(contentsOf: url)
        guard let json = String(data: data, encoding: .utf8) else {
            throw LoadError.decodingFailed(message: "model_registry.json is not valid UTF-8")
        }
        self.registry = ModelRegistry.companion.parse(jsonString: json)
    }
}
