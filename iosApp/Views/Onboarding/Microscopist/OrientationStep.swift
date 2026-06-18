import SwiftUI

/// Phase 2 step 4 (spec §10): three-page orientation. Skippable.
/// Last page advances to operational state via
/// `OnboardingState.finishOrientation()`.
struct OrientationStep: View {

    @Environment(OnboardingState.self) private var onboarding

    @State private var page: Int = 0

    private let pages: [OrientationPage] = [
        OrientationPage(
            symbol: "viewfinder",
            title: "How to capture",
            body: "Tap the Capture button on the Home tab to take a photo through the microscope. The app classifies each cell in 50–300 ms and shows the result inline."
        ),
        OrientationPage(
            symbol: "exclamationmark.triangle",
            title: "How to override",
            body: "If you disagree with the model, tap Override next to the prediction. Pick the corrected verdict and a reason. The original prediction is preserved in the audit log."
        ),
        OrientationPage(
            symbol: "lock.fill",
            title: "How to lock",
            body: "Tap the lock icon in the corner — or just background the app — to lock the session. Re-unlock with your biometric. Auto-logout fires after the timeout the admin configured."
        ),
    ]

    var body: some View {
        WizardStepContainer(
            title: "Quick orientation",
            subtitle: "Three short cards. Swipe to navigate or skip to the end.",
            stepIndicator: "Step 4 of 4",
            primary: .init(title: page == pages.count - 1 ? "Begin screening" : "Next") {
                if page == pages.count - 1 {
                    onboarding.finishOrientation()
                } else {
                    page += 1
                }
            },
            secondary: .init(title: "Skip orientation") {
                onboarding.finishOrientation()
            }
        ) {
            TabView(selection: $page) {
                ForEach(Array(pages.enumerated()), id: \.offset) { idx, item in
                    VStack(alignment: .leading, spacing: 16) {
                        Image(systemName: item.symbol)
                            .font(.system(size: 64))
                            .foregroundStyle(Color.accentColor)
                            .padding(.top, 16)
                        Text(item.title)
                            .font(.title2.weight(.semibold))
                        Text(item.body)
                            .font(.body)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Spacer()
                    }
                    .padding(.leading, 4)
                    .padding(.trailing, 4)
                    .tag(idx)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .always))
            .frame(minHeight: 320)
        }
    }
}

private struct OrientationPage {
    let symbol: String
    let title: String
    let body: String
}
