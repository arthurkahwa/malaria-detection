import SwiftUI

/// Free-text session label editor with the PII warning copy required by
/// spec §13. Labels are max 20 chars, plain ASCII only (rejects emoji
/// and non-ASCII to keep exports clean).
struct SessionRelabelView: View {

    let sessionId: String

    @Environment(\.dismiss) private var dismiss
    @Environment(PredictionStore.self) private var predictionStore

    @State private var label: String = ""
    @State private var errorText: String? = nil

    static let maxLength = 20

    /// True iff `s` is non-empty, <= 20 chars, plain ASCII, and contains
    /// no control characters. Emoji are non-ASCII by definition so they
    /// fail this check.
    static func isValidLabel(_ s: String) -> Bool {
        guard !s.isEmpty, s.count <= maxLength else { return false }
        for scalar in s.unicodeScalars {
            if !scalar.isASCII { return false }
            // Reject ASCII control chars; keep printable 0x20..=0x7E.
            if scalar.value < 0x20 || scalar.value == 0x7F { return false }
        }
        return true
    }

    var canSave: Bool { Self.isValidLabel(label) }

    var body: some View {
        Form {
            Section {
                Text("Labels appear in exports and audit reports. Do not include identifying information per your clinic's privacy policy.")
                    .font(.callout)
                    .foregroundStyle(.orange)
            }

            Section("Label") {
                TextField("Session label", text: $label)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled(true)
                    .onChange(of: label) { _, newValue in
                        if newValue.count > Self.maxLength {
                            label = String(newValue.prefix(Self.maxLength))
                        }
                    }
                HStack {
                    Text("\(label.count)/\(Self.maxLength) characters")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if !label.isEmpty && !Self.isValidLabel(label) {
                        Text("ASCII only, no emoji")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            }

            if let errorText {
                Section {
                    Text(errorText)
                        .foregroundStyle(.red)
                }
            }
        }
        .navigationTitle("Edit label")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel") { dismiss() }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") {
                    save()
                }
                .disabled(!canSave)
            }
        }
    }

    private func save() {
        guard Self.isValidLabel(label) else { return }
        do {
            try predictionStore.relabel(sessionId: sessionId, label: label)
            dismiss()
        } catch {
            errorText = error.localizedDescription
        }
    }
}
