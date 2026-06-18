import SwiftUI
@preconcurrency import Shared

/// Models section (spec §11). Three groups:
///
/// 1. Bundled — every `bundled = true` entry from `model_registry.json`.
///    Marked "Available offline" with a check.
/// 2. Downloaded — models currently cached in the application support directory.
/// 3. Available — every other non-bundled entry, downloadable from HF.
///
/// A "Clear all caches" action is enabled when at least one model is downloaded.
struct ModelsSection: View {

    @Environment(ModelRegistryService.self) private var modelRegistry
    @Environment(ModelDownloadService.self) private var modelDownloadService
    @Environment(SettingsStore.self) private var settings

    @State private var showClearConfirmation = false

    var body: some View {
        Group {
            // MARK: Bundled
            Section {
                ForEach(bundledEntries, id: \.id) { entry in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(entry.displayName)
                            Text(entry.id)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Label("Available offline", systemImage: "checkmark.circle.fill")
                            .labelStyle(.iconOnly)
                            .foregroundStyle(.green)
                    }
                }
            } header: {
                Text("Bundled")
            } footer: {
                Text("Bundled models ship inside the app and run without network access.")
            }

            // MARK: Downloaded
            Section {
                if downloadedEntries.isEmpty {
                    Text("No downloaded models")
                        .foregroundStyle(.secondary)
                        .italic()
                } else {
                    ForEach(downloadedEntries, id: \.id) { entry in
                        downloadedRow(entry: entry)
                    }
                }
            } header: {
                HStack {
                    Text("Downloaded")
                    if modelDownloadService.downloadedModelCount > 0 {
                        Text("\(modelDownloadService.downloadedModelCount)")
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Color.blue.opacity(0.2))
                            .foregroundStyle(.blue)
                            .clipShape(Capsule())
                    }
                }
            } footer: {
                Text("Downloaded models are stored in the app sandbox and survive app restarts.")
            }

            // MARK: Available
            Section {
                ForEach(availableEntries, id: \.id) { entry in
                    availableRow(entry: entry)
                }
            } header: {
                Text("Available")
            } footer: {
                cacheFooter
            }
        }
        .confirmationDialog(
            "Clear all downloaded models?",
            isPresented: $showClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear all caches", role: .destructive) {
                modelDownloadService.clearAllCaches(settings: settings)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will delete all downloaded models from the app sandbox. The bundled BNLeaky_Keras model will remain available.")
        }
    }

    // MARK: - Computed entry lists

    private var bundledEntries: [ModelRegistryEntry] {
        modelRegistry.registry.all().filter { $0.bundled }
    }

    /// Non-bundled entries that are in .downloaded state.
    private var downloadedEntries: [ModelRegistryEntry] {
        modelRegistry.registry.all().filter { entry in
            !entry.bundled &&
            modelDownloadService.downloadStates[entry.id] == .downloaded
        }
    }

    /// Non-bundled entries that are NOT in .downloaded state (available for download).
    private var availableEntries: [ModelRegistryEntry] {
        modelRegistry.registry.all().filter { entry in
            guard !entry.bundled else { return false }
            switch modelDownloadService.downloadStates[entry.id] ?? .notDownloaded {
            case .downloaded: return false
            default: return true
            }
        }
    }

    // MARK: - Row builders

    @ViewBuilder
    private func downloadedRow(entry: ModelRegistryEntry) -> some View {
        HStack {
            VStack(alignment: .leading) {
                Text(entry.displayName)
                Text(entry.id)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Label("Downloaded", systemImage: "arrow.down.circle.fill")
                .labelStyle(.iconOnly)
                .foregroundStyle(.blue)
            Button {
                modelDownloadService.deleteModel(modelId: entry.id)
            } label: {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
        }
    }

    @ViewBuilder
    private func availableRow(entry: ModelRegistryEntry) -> some View {
        let state = modelDownloadService.downloadStates[entry.id] ?? .notDownloaded
        switch state {
        case .notDownloaded:
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.displayName)
                    Text(String(format: "%.1f MB • Requires internet", entry.iosFileSizeMb))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Download") {
                    Task { await modelDownloadService.download(entry: entry) }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

        case .downloading(let progress):
            VStack(alignment: .leading, spacing: 6) {
                Text(entry.displayName)
                ProgressView(value: progress)
                let receivedMb = entry.iosFileSizeMb * progress
                Text(String(format: "%.1f MB of %.1f MB", receivedMb, entry.iosFileSizeMb))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

        case .compiling:
            VStack(alignment: .leading, spacing: 6) {
                Text(entry.displayName)
                ProgressView()
                Text("Compiling…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

        case .downloaded:
            // Should not appear here (covered by downloadedEntries), but handle defensively.
            EmptyView()

        case .failed(let message):
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.displayName)
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                Spacer()
                Button("Retry") {
                    Task { await modelDownloadService.download(entry: entry) }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(.orange)
            }
        }
    }

    // MARK: - Footer

    private var cacheSizeMb: Double {
        downloadedEntries.reduce(0) { $0 + $1.iosFileSizeMb }
    }

    private var cacheFooter: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(String(format: "Total cache size: %.1f MB", cacheSizeMb))
                .font(.caption)
            Button(role: .destructive) {
                showClearConfirmation = true
            } label: {
                Text("Clear all caches")
            }
            .disabled(!modelDownloadService.hasDownloadedModels)
        }
    }
}
