import SwiftUI
@preconcurrency import Shared

/// Two-tap live override modal during active screening (spec §12 live
/// override). Screen 1 picks the corrected verdict; Screen 2 picks the
/// canonical reason. Tapping a reason writes the override and dismisses
/// the sheet — no biometric, no notes, no initials per spec §12 (the
/// review-override flow in History is the path for fuller context).
///
/// Round-trip target: ~3 seconds. The microscopist returns to the
/// camera ready for the next cell.
struct LiveOverrideSheet: View {

    let prediction: Prediction

    @Environment(\.dismiss) private var dismiss
    @Environment(PredictionStore.self) private var predictionStore

    enum Verdict: String {
        case parasitized = "Parasitized"
        case uninfected = "Uninfected"
    }

    enum Step {
        case verdict
        case reason(Verdict)
    }

    @State private var step: Step = .verdict
    @State private var errorText: String? = nil

    private var probabilityPercent: Int {
        Int((prediction.parasitizedProb * 100).rounded())
    }

    var body: some View {
        NavigationStack {
            Group {
                switch step {
                case .verdict:
                    verdictScreen
                case .reason(let verdict):
                    reasonScreen(verdict: verdict)
                }
            }
            .navigationTitle(navigationTitleText)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private var navigationTitleText: String {
        switch step {
        case .verdict: return "Override"
        case .reason: return "Reason"
        }
    }

    // MARK: - Screen 1 — verdict picker

    private var verdictScreen: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text("The model said: \(prediction.label) (\(probabilityPercent)%)")
                .font(.headline)
                .padding(.top, 24)

            Text("Override to:")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            VStack(spacing: 16) {
                verdictButton(.parasitized)
                verdictButton(.uninfected)
            }

            if let errorText {
                Text(errorText)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Spacer()
        }
        .padding(.leading, 24)
        .padding(.trailing, 24)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func verdictButton(_ verdict: Verdict) -> some View {
        Button {
            step = .reason(verdict)
        } label: {
            Text(verdict.rawValue)
                .font(.title3.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 18)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .accessibilityIdentifier("liveOverrideVerdict_\(verdict.rawValue)")
    }

    // MARK: - Screen 2 — reason picker

    private func reasonScreen(verdict: Verdict) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Reason:")
                .font(.headline)
                .padding(.top, 24)

            ForEach(Shared.OverrideReason.entries, id: \.canonical) { reason in
                Button {
                    Task { await commit(verdict: verdict, reason: reason) }
                } label: {
                    HStack {
                        Text(ReviewOverrideView.displayLabel(for: reason))
                            .font(.body)
                        Spacer()
                    }
                    .padding(.vertical, 14)
                    .padding(.leading, 16)
                    .padding(.trailing, 16)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .accessibilityIdentifier("liveOverrideReason_\(reason.canonical)")
            }

            if let errorText {
                Text(errorText)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Spacer()
        }
        .padding(.leading, 24)
        .padding(.trailing, 24)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Commit

    private func commit(verdict: Verdict, reason: Shared.OverrideReason) async {
        do {
            try predictionStore.override(
                prediction,
                verdict: verdict.rawValue,
                context: Shared.OverrideContext.live.canonical,
                reason: reason.canonical,
                notes: nil,
                actorInitials: nil,
                contextReviewed: nil
            )
            dismiss()
        } catch {
            errorText = error.localizedDescription
        }
    }
}
