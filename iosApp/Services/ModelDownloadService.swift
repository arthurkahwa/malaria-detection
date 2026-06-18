import CoreML
import Foundation
import Observation
@preconcurrency import Shared

// MARK: - State

enum ModelDownloadState: Equatable {
    case notDownloaded
    case downloading(progress: Double) // 0.0–1.0
    case compiling
    case downloaded
    case failed(String)
}

// MARK: - Service

/// Downloads, compiles and caches Hugging Face–hosted `.mlpackage` models.
///
/// Each `.mlpackage` is a directory; HF serves individual files. We fetch
/// three files per model, reconstruct the directory tree in a temp folder,
/// compile it with `MLModel.compileModel(at:)`, then move the resulting
/// `.mlmodelc` bundle into `applicationSupportDirectory/MalariaDetector/models/`.
///
/// The Kotlin-side `CoreMLClassifier` checks that directory first before
/// falling back to the main bundle (which only bundles `BNLeaky_Keras`).
@Observable
@MainActor
final class ModelDownloadService {

    // MARK: - Public state

    /// Download state keyed by model id.
    private(set) var downloadStates: [String: ModelDownloadState] = [:]

    // MARK: - Private

    private let cacheDir: URL
    private var activeTasks: [String: Task<Void, Never>] = [:]

    // MARK: - Init

    init() {
        let appSupport = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
        let dir = appSupport
            .appendingPathComponent("MalariaDetector")
            .appendingPathComponent("models")
        self.cacheDir = dir

        // Create directory if needed (non-throwing; failure is tolerated).
        try? FileManager.default.createDirectory(
            at: dir,
            withIntermediateDirectories: true
        )

        // Scan existing compiled bundles and mark as downloaded.
        if let contents = try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: nil
        ) {
            for url in contents where url.pathExtension == "mlmodelc" {
                let modelId = url.deletingPathExtension().lastPathComponent
                downloadStates[modelId] = .downloaded
            }
        }
    }

    // MARK: - Public API

    /// Start downloading a model. Idempotent — ignores calls when already
    /// downloading or already downloaded.
    func download(entry: ModelRegistryEntry) async {
        let modelId = entry.id

        switch downloadStates[modelId] {
        case .downloading, .downloaded:
            return
        default:
            break
        }

        guard let iosPath = entry.iosPath,
              let hfRepo = entry.huggingfaceRepo else {
            downloadStates[modelId] = .failed("Missing iosPath or huggingfaceRepo in registry")
            return
        }

        let repo = hfRepo.replacingOccurrences(of: "{maintainer}", with: "arthurkahwa")
        let baseURL = "https://huggingface.co/\(repo)/resolve/main"

        downloadStates[modelId] = .downloading(progress: 0)

        let task = Task {
            await performDownload(
                modelId: modelId,
                iosPath: iosPath,
                baseURL: baseURL,
                fileSizeEstimateMb: entry.iosFileSizeMb
            )
        }
        activeTasks[modelId] = task
    }

    /// Delete the cached compiled model. Also cancels an in-progress download.
    func deleteModel(modelId: String) {
        activeTasks[modelId]?.cancel()
        activeTasks[modelId] = nil
        let url = compiledModelURL(for: modelId)
        try? FileManager.default.removeItem(at: url)
        downloadStates[modelId] = .notDownloaded
    }

    /// Returns the URL to a compiled `.mlmodelc` if it exists on disk.
    func cachedModelURL(for modelId: String) -> URL? {
        let url = compiledModelURL(for: modelId)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return url
    }

    /// Delete all downloaded models from the cache directory. Resets
    /// `SettingsStore.defaultModelId` to `BNLeaky_Keras` if the current
    /// default is a cached (non-bundled) model.
    func clearAllCaches(settings: SettingsStore) {
        // Cancel active downloads.
        for task in activeTasks.values { task.cancel() }
        activeTasks.removeAll()

        // Remove every .mlmodelc in the cache directory.
        if let contents = try? FileManager.default.contentsOfDirectory(
            at: cacheDir,
            includingPropertiesForKeys: nil
        ) {
            for url in contents where url.pathExtension == "mlmodelc" {
                try? FileManager.default.removeItem(at: url)
            }
        }

        // Reset UI state.
        for key in downloadStates.keys {
            downloadStates[key] = .notDownloaded
        }

        // Reset default model if it was cached.
        if settings.defaultModelId != "BNLeaky_Keras" {
            settings.updateDefaultModel("BNLeaky_Keras")
        }
    }

    /// True when at least one model is in the `.downloaded` state.
    var hasDownloadedModels: Bool {
        downloadStates.values.contains(.downloaded)
    }

    /// Count of models currently in `.downloaded` state.
    var downloadedModelCount: Int {
        downloadStates.values.filter { $0 == .downloaded }.count
    }

    // MARK: - Private helpers

    private func compiledModelURL(for modelId: String) -> URL {
        cacheDir.appendingPathComponent("\(modelId).mlmodelc")
    }

    /// The three relative paths inside an `.mlpackage` that HF must serve.
    private func filePaths(for iosPath: String) -> [(relative: String, isWeightBin: Bool)] {
        [
            ("\(iosPath)/Manifest.json", false),
            ("\(iosPath)/Data/com.apple.CoreML/model.mlmodel", false),
            ("\(iosPath)/Data/com.apple.CoreML/weights/weight.bin", true),
        ]
    }

    // MARK: - Core download logic

    private func performDownload(
        modelId: String,
        iosPath: String,
        baseURL: String,
        fileSizeEstimateMb: Double
    ) async {
        let tmpBase = FileManager.default.temporaryDirectory
            .appendingPathComponent("malariaDownloads")
            .appendingPathComponent(modelId)

        // Clean any stale temp directory.
        try? FileManager.default.removeItem(at: tmpBase)

        do {
            try FileManager.default.createDirectory(
                at: tmpBase,
                withIntermediateDirectories: true
            )

            let files = filePaths(for: iosPath)
            let estimatedWeightBytes = Int64(fileSizeEstimateMb * 1_024 * 1_024)

            for (relativePath, isWeightBin) in files {
                if Task.isCancelled {
                    await MainActor.run { downloadStates[modelId] = .notDownloaded }
                    return
                }

                guard let url = URL(string: "\(baseURL)/\(relativePath)") else {
                    await MainActor.run {
                        downloadStates[modelId] = .failed("Invalid URL for \(relativePath)")
                    }
                    return
                }

                let destURL = tmpBase.appendingPathComponent(relativePath)
                try FileManager.default.createDirectory(
                    at: destURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )

                let request = URLRequest(url: url)
                let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)

                guard let httpResponse = response as? HTTPURLResponse else {
                    await MainActor.run {
                        downloadStates[modelId] = .failed("No HTTP response for \(relativePath)")
                    }
                    return
                }

                if httpResponse.statusCode == 404 {
                    await MainActor.run {
                        downloadStates[modelId] = .failed("Model not yet available on Hugging Face")
                    }
                    return
                }

                guard (200..<300).contains(httpResponse.statusCode) else {
                    await MainActor.run {
                        downloadStates[modelId] = .failed("HTTP \(httpResponse.statusCode) for \(relativePath)")
                    }
                    return
                }

                // Use Content-Length if available; otherwise fall back to file-size estimate.
                let expectedBytes: Int64
                if let lengthStr = httpResponse.value(forHTTPHeaderField: "Content-Length"),
                   let length = Int64(lengthStr), length > 0 {
                    expectedBytes = length
                } else if isWeightBin {
                    expectedBytes = estimatedWeightBytes
                } else {
                    expectedBytes = 0
                }

                // Stream to disk.
                var receivedBytes: Int64 = 0
                var buffer = Data()
                buffer.reserveCapacity(isWeightBin ? 256 * 1024 : 64 * 1024)

                let outputStream = OutputStream(url: destURL, append: false)
                outputStream?.open()
                defer { outputStream?.close() }

                for try await byte in asyncBytes {
                    if Task.isCancelled { break }
                    buffer.append(byte)
                    receivedBytes += 1

                    // Flush buffer every 256 KB for weight.bin, 64 KB otherwise.
                    let flushSize = isWeightBin ? 256 * 1024 : 64 * 1024
                    if buffer.count >= flushSize {
                        let written = buffer.withUnsafeBytes { ptr in
                            outputStream?.write(
                                ptr.bindMemory(to: UInt8.self).baseAddress!,
                                maxLength: buffer.count
                            ) ?? 0
                        }
                        if written < 0 {
                            throw URLError(.cannotWriteToFile)
                        }
                        buffer.removeAll(keepingCapacity: true)
                    }

                    // Update progress only for the weight.bin (the big file).
                    if isWeightBin && expectedBytes > 0 {
                        let progress = min(Double(receivedBytes) / Double(expectedBytes), 1.0)
                        // Throttle UI updates to every ~0.5% to avoid flooding MainActor.
                        if receivedBytes % 50_000 == 0 {
                            let p = progress
                            await MainActor.run { downloadStates[modelId] = .downloading(progress: p) }
                        }
                    }
                }

                // Flush remaining bytes.
                if !buffer.isEmpty {
                    buffer.withUnsafeBytes { ptr in
                        _ = outputStream?.write(
                            ptr.bindMemory(to: UInt8.self).baseAddress!,
                            maxLength: buffer.count
                        )
                    }
                }

                if Task.isCancelled {
                    await MainActor.run { downloadStates[modelId] = .notDownloaded }
                    try? FileManager.default.removeItem(at: tmpBase)
                    return
                }
            }

            // All files downloaded — compile.
            await MainActor.run { downloadStates[modelId] = .compiling }

            let packageURL = tmpBase.appendingPathComponent(iosPath)
            let compiledTmpURL = try await MLModel.compileModel(at: packageURL)

            let destURL = compiledModelURL(for: modelId)
            // Remove any stale compiled bundle.
            try? FileManager.default.removeItem(at: destURL)
            try FileManager.default.moveItem(at: compiledTmpURL, to: destURL)

            // Clean up temp directory.
            try? FileManager.default.removeItem(at: tmpBase)

            await MainActor.run { downloadStates[modelId] = .downloaded }

        } catch is CancellationError {
            try? FileManager.default.removeItem(at: tmpBase)
            await MainActor.run { downloadStates[modelId] = .notDownloaded }
        } catch {
            try? FileManager.default.removeItem(at: tmpBase)
            await MainActor.run {
                downloadStates[modelId] = .failed(error.localizedDescription)
            }
        }
    }
}
