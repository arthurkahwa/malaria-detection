import SwiftUI

/// Decision threshold slider (spec §10 step 6 and §11 inference policy).
///
/// Range 0.0–1.0 in 0.05 increments. Shows the live value and the spec's
/// trade-off explanatory caption so the admin understands what they're
/// dialing in.
struct ThresholdSlider: View {

    @Binding var threshold: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Decision threshold")
                    .font(.headline)
                Spacer()
                Text(String(format: "%.2f", threshold))
                    .font(.body.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Slider(value: $threshold, in: 0.0...1.0, step: 0.05) {
                Text("Decision threshold")
            } minimumValueLabel: {
                Text("0.0").font(.caption)
            } maximumValueLabel: {
                Text("1.0").font(.caption)
            }

            Text("Lower values flag more cells as parasitized (more false positives, fewer missed parasites). Higher values flag fewer cells (more missed parasites, fewer false positives). 0.30 is the v1 default.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
