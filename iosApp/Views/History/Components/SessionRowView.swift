import SwiftUI
@preconcurrency import Shared

/// One row in the Sessions list. Also embedded in PredictionDetailView's
/// "Session" section as a navigable affordance.
struct SessionRowView: View {
    let stats: SessionStats

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(stats.displayLabel)
                    .font(.headline)
                    .lineLimit(1)
                Spacer()
                Text("\(stats.count) cells")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 12) {
                statBadge(label: "Parasitized", value: "\(stats.parasitizedCount)")
                statBadge(label: "Gray zone", value: "\(stats.grayZoneCount)")
                statBadge(label: "Mean p", value: stats.meanParasitizedFormatted)
            }
            Text(stats.dateRangeLabel)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }

    private func statBadge(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.semibold))
        }
    }
}

/// Aggregated stats for one session, computed in Swift from a list of
/// predictions. Session-stats live in shared Kotlin in v1.1 (spec §13)
/// but we keep them client-side for Phase 10 — no read-path mutation
/// needed in repos.
struct SessionStats: Identifiable, Hashable {
    let sessionId: String
    let sessionLabel: String?
    let count: Int
    let parasitizedCount: Int
    let grayZoneCount: Int
    let meanParasitizedProb: Double
    let earliest: Date
    let latest: Date

    var id: String { sessionId }

    var displayLabel: String {
        if let label = sessionLabel, !label.isEmpty {
            return label
        }
        return "Session \(String(sessionId.prefix(8)))"
    }

    var meanParasitizedFormatted: String {
        "\(Int((meanParasitizedProb * 100).rounded()))%"
    }

    var dateRangeLabel: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        if Calendar.current.isDate(earliest, inSameDayAs: latest) {
            let dayFmt = DateFormatter()
            dayFmt.dateStyle = .medium
            let timeFmt = DateFormatter()
            timeFmt.timeStyle = .short
            return "\(dayFmt.string(from: earliest)) · \(timeFmt.string(from: earliest))–\(timeFmt.string(from: latest))"
        } else {
            return "\(formatter.string(from: earliest)) – \(formatter.string(from: latest))"
        }
    }

    /// Compute stats from an arbitrary list of predictions in one session.
    /// Uses `Threshold.shared.GRAY_ZONE_*` for gray-zone classification so
    /// the count matches the rest of the app exactly.
    static func from(predictions: [Prediction]) -> SessionStats? {
        guard let first = predictions.first else { return nil }
        let sortedByTime = predictions.sorted { $0.timestamp < $1.timestamp }
        let count = predictions.count
        let parasitizedCount = predictions.filter { $0.label == "Parasitized" }.count
        let low = Double(Threshold.shared.GRAY_ZONE_LOW)
        let high = Double(Threshold.shared.GRAY_ZONE_HIGH)
        let grayZoneCount = predictions.filter { $0.parasitizedProb >= low && $0.parasitizedProb <= high }.count
        let meanProb = predictions.map(\.parasitizedProb).reduce(0, +) / Double(count)
        return SessionStats(
            sessionId: first.sessionId,
            sessionLabel: first.sessionLabel,
            count: count,
            parasitizedCount: parasitizedCount,
            grayZoneCount: grayZoneCount,
            meanParasitizedProb: meanProb,
            earliest: sortedByTime.first?.timestamp ?? first.timestamp,
            latest: sortedByTime.last?.timestamp ?? first.timestamp
        )
    }

    /// Group a flat prediction list into per-session stats, newest session
    /// first (latest timestamp).
    static func grouped(_ predictions: [Prediction]) -> [SessionStats] {
        let groups = Dictionary(grouping: predictions, by: \.sessionId)
        return groups.compactMap { _, list in SessionStats.from(predictions: list) }
            .sorted { $0.latest > $1.latest }
    }
}
