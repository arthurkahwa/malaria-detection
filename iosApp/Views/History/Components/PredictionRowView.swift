import SwiftUI

/// Row layout used in RecentPredictions, FlaggedForReview, and the
/// in-session list inside SessionDetail. Surfaces verdict, probability,
/// model id, override status, and a relative + absolute timestamp.
struct PredictionRowView: View {
    let prediction: Prediction

    private var probabilityPercent: Int {
        Int((prediction.parasitizedProb * 100).rounded())
    }

    private var relativeTimestamp: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter.localizedString(for: prediction.timestamp, relativeTo: Date())
    }

    private var absoluteTimestamp: String {
        prediction.timestamp.formatted(date: .abbreviated, time: .shortened)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(prediction.label)
                    .font(.headline)
                Text("\(probabilityPercent)%")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                RiskBandIndicator(parasitizedProb: prediction.parasitizedProb)
            }
            HStack(spacing: 8) {
                Text(relativeTimestamp)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("·")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(absoluteTimestamp)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                badge(text: prediction.modelId, color: .blue)
                if prediction.clinicianOverride != nil {
                    badge(text: "Overridden", color: .purple)
                }
                if prediction.duplicateOfId != nil {
                    badge(text: "Duplicate", color: .gray)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func badge(text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.leading, 6)
            .padding(.trailing, 6)
            .padding(.top, 2)
            .padding(.bottom, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}
