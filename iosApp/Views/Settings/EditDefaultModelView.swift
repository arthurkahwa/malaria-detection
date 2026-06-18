import SwiftUI
@preconcurrency import Shared

/// Edit the default model. Bundled models are always selectable. Non-bundled
/// models can be downloaded from Hugging Face; once downloaded they become
/// selectable and can be deleted.
struct EditDefaultModelView: View {

    @Environment(\.dismiss) private var dismiss
    @Environment(AuthGate.self) private var authGate
    @Environment(SettingsStore.self) private var settings
    @Environment(ModelRegistryService.self) private var modelRegistry
    @Environment(ModelDownloadService.self) private var modelDownloadService

    @State private var selectedId: String = "BNLeaky_Keras"
    @State private var isSaving: Bool = false
    @State private var errorText: String?

    private var entries: [ModelRegistryEntry] {
        modelRegistry.registry.all()
    }

    var body: some View {
        Form {
            Section("Default model") {
                ForEach(entries, id: \.id) { entry in
                    modelRow(entry: entry)
                }
            }
            if let errorText {
                Section { Text(errorText).foregroundStyle(.red) }
            }
        }
        .navigationTitle("Default model")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") {
                    Task { await save() }
                }
                .disabled(isSaving)
            }
        }
        .onAppear {
            selectedId = settings.defaultModelId
        }
    }

    // MARK: - Row builder

    @ViewBuilder
    private func modelRow(entry: ModelRegistryEntry) -> some View {
        if entry.bundled {
            // Bundled — always selectable.
            Button {
                selectedId = entry.id
            } label: {
                HStack {
                    VStack(alignment: .leading) {
                        Text(entry.displayName)
                        Text("Bundled")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("Bundled")
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.15))
                        .foregroundStyle(.green)
                        .clipShape(Capsule())
                    if selectedId == entry.id {
                        Image(systemName: "checkmark")
                            .foregroundStyle(Color.accentColor)
                    }
                }
            }
        } else {
            // Non-bundled — state-driven row.
            let state = modelDownloadService.downloadStates[entry.id] ?? .notDownloaded
            nonBundledRow(entry: entry, state: state)
        }
    }

    @ViewBuilder
    private func nonBundledRow(entry: ModelRegistryEntry, state: ModelDownloadState) -> some View {
        switch state {
        case .notDownloaded:
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.displayName)
                    Text(String(format: "%.1f MB", entry.iosFileSizeMb))
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
            Button {
                selectedId = entry.id
            } label: {
                HStack {
                    VStack(alignment: .leading) {
                        Text(entry.displayName)
                        Text("Downloaded")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("Downloaded")
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.blue.opacity(0.15))
                        .foregroundStyle(.blue)
                        .clipShape(Capsule())
                    if selectedId == entry.id {
                        Image(systemName: "checkmark")
                            .foregroundStyle(Color.accentColor)
                    }
                    Button {
                        // If this model was the selected one, fall back to bundled default.
                        if selectedId == entry.id {
                            selectedId = "BNLeaky_Keras"
                        }
                        modelDownloadService.deleteModel(modelId: entry.id)
                    } label: {
                        Image(systemName: "trash")
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                }
            }

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

    // MARK: - Save

    private func save() async {
        errorText = nil
        isSaving = true
        defer { isSaving = false }

        do {
            try await authGate.unlock(reason: "Confirm default-model change")
        } catch {
            errorText = error.localizedDescription
            return
        }

        settings.updateDefaultModel(selectedId)
        dismiss()
    }
}
