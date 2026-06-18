import SwiftUI

/// Edit the decision threshold (admin only). Fresh biometric before
/// `threshold_changed` audit entry per spec §9 + §11.
struct EditThresholdView: View {

    @Environment(\.dismiss) private var dismiss
    @Environment(AuthGate.self) private var authGate
    @Environment(SettingsStore.self) private var settings

    @State private var threshold: Double = 0.3
    @State private var isSaving: Bool = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading) {
                    Slider(value: $threshold, in: 0.0...1.0, step: 0.01)
                    Text(String(format: "%.2f", threshold))
                        .font(.system(.body, design: .monospaced))
                }
            } footer: {
                Text("Lower threshold → more false positives, fewer false negatives. Higher threshold → fewer false positives, more false negatives.")
            }
            if let errorText {
                Section { Text(errorText).foregroundStyle(.red) }
            }
        }
        .navigationTitle("Edit threshold")
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
            threshold = settings.threshold
        }
    }

    private func save() async {
        errorText = nil
        isSaving = true
        defer { isSaving = false }

        do {
            try await authGate.unlock(reason: "Confirm threshold change")
        } catch {
            errorText = error.localizedDescription
            return
        }

        settings.updateThreshold(threshold)
        dismiss()
    }
}
