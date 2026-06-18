import SwiftUI
@preconcurrency import Shared

/// Visual risk-band indicator for a parasitized probability.
///
/// Per spec §11: low (< GRAY_ZONE_LOW) renders as gray, gray-zone
/// (`[GRAY_ZONE_LOW, GRAY_ZONE_HIGH]`) as amber, and high
/// (> GRAY_ZONE_HIGH) as red. Constants live in shared `Threshold`
/// so both platforms agree on the band boundaries.
struct RiskBandIndicator: View {
    let parasitizedProb: Double

    enum Band: String {
        case low
        case grayZone
        case high

        var color: Color {
            switch self {
            case .low: .gray
            case .grayZone: .orange
            case .high: .red
            }
        }

        var label: String {
            switch self {
            case .low: "Low"
            case .grayZone: "Gray zone"
            case .high: "High"
            }
        }
    }

    static func band(for parasitizedProb: Double) -> Band {
        let low = Double(Threshold.shared.GRAY_ZONE_LOW)
        let high = Double(Threshold.shared.GRAY_ZONE_HIGH)
        if parasitizedProb < low { return .low }
        if parasitizedProb > high { return .high }
        return .grayZone
    }

    private var band: Band { Self.band(for: parasitizedProb) }

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(band.color)
                .frame(width: 10, height: 10)
            Text(band.label)
                .font(.caption.weight(.medium))
                .foregroundStyle(band.color)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Risk band: \(band.label)")
    }
}
