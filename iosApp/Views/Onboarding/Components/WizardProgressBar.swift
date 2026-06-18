import SwiftUI

/// Tiny "Step N of M" indicator. Kept separate from the container so a
/// future redesign (a dot-grid, a segmented bar) drops in by swapping
/// this one View.
struct WizardProgressBar: View {
    let currentStep: Int
    let totalSteps: Int

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<totalSteps, id: \.self) { idx in
                Capsule()
                    .fill(idx < currentStep ? Color.accentColor : Color.secondary.opacity(0.3))
                    .frame(height: 4)
            }
        }
        .accessibilityLabel(Text("Step \(currentStep) of \(totalSteps)"))
    }
}
