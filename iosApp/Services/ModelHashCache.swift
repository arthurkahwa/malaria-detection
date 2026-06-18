import CryptoKit
import Foundation

/// Computes and caches the SHA-256 hash of a bundled CoreML model.
///
/// The compiled `.mlmodelc` bundle is a directory; we hash all regular
/// files inside it sorted by path so the digest is deterministic.
/// The result is cached after the first call — the model file doesn't
/// change at runtime, so repeated lookups are free.
enum ModelHashCache {
    nonisolated(unsafe) private static var cache: [String: String] = [:]
    private static let lock = NSLock()

    static func hash(for modelId: String) -> String {
        lock.lock()
        defer { lock.unlock() }
        if let cached = cache[modelId] { return cached }
        let h = computeHash(modelId: modelId) ?? modelId
        cache[modelId] = h
        return h
    }

    private static func computeHash(modelId: String) -> String? {
        guard let root = Bundle.main.url(
            forResource: "Malaria_\(modelId)",
            withExtension: "mlmodelc"
        ) else { return nil }

        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        let files = enumerator
            .compactMap { $0 as? URL }
            .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true }
            .sorted { $0.path < $1.path }

        var hasher = SHA256()
        for url in files {
            guard let data = try? Data(contentsOf: url) else { continue }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
