import SwiftData
import SwiftUI

/// Picker for choosing the original prediction this one duplicates.
/// Per spec §13: candidates are the last 50 predictions in the same
/// session, excluding the duplicate itself. Confirmation is required.
struct MarkAsDuplicateView: View {

    let duplicate: Prediction

    @Environment(\.dismiss) private var dismiss
    @Environment(PredictionStore.self) private var predictionStore

    @Query private var candidates: [Prediction]

    @State private var selectedId: String? = nil
    @State private var showConfirm = false
    @State private var errorText: String? = nil

    init(duplicate: Prediction) {
        self.duplicate = duplicate
        let sessionId = duplicate.sessionId
        let duplicateId = duplicate.id
        var descriptor = FetchDescriptor<Prediction>(
            predicate: #Predicate<Prediction> {
                $0.sessionId == sessionId && $0.id != duplicateId
            },
            sortBy: [SortDescriptor(\Prediction.timestamp, order: .reverse)]
        )
        descriptor.fetchLimit = 50
        _candidates = Query(descriptor)
    }

    private var selectedCandidate: Prediction? {
        guard let selectedId else { return nil }
        return candidates.first { $0.id == selectedId }
    }

    var body: some View {
        List {
            Section {
                Text("Choose the original prediction that this one duplicates. The duplicate stays in storage — no data is deleted.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            if candidates.isEmpty {
                Section {
                    Text("No other predictions in this session.")
                        .foregroundStyle(.secondary)
                }
            } else {
                Section("Candidates") {
                    ForEach(candidates) { candidate in
                        Button {
                            selectedId = candidate.id
                            showConfirm = true
                        } label: {
                            PredictionRowView(prediction: candidate)
                        }
                        .buttonStyle(.plain)
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
        .navigationTitle("Mark as duplicate")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel") { dismiss() }
            }
        }
        .alert("Mark as duplicate?", isPresented: $showConfirm) {
            Button("Cancel", role: .cancel) {
                selectedId = nil
            }
            Button("Mark") {
                confirmMark()
            }
        } message: {
            if let candidate = selectedCandidate {
                Text("This prediction will be linked as a duplicate of the one captured at \(candidate.timestamp.formatted(date: .abbreviated, time: .shortened)).")
            }
        }
    }

    private func confirmMark() {
        guard let candidate = selectedCandidate else { return }
        do {
            try predictionStore.markAsDuplicate(duplicate, of: candidate.id)
            dismiss()
        } catch {
            errorText = error.localizedDescription
        }
    }
}
