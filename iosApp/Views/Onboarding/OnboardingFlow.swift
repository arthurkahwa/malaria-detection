import SwiftUI

/// Top-level onboarding coordinator (spec §10).
///
/// Reads `OnboardingState.phase` and routes to the matching sub-flow:
/// admin provisioning (Phase 1), microscopist claim (Phase 2), or — when
/// the phase is `.complete` — renders nothing so the composition root
/// shows `RootView` instead.
///
/// The composition root (`MalariaDetectorApp`) decides which view to
/// mount based on the same `phase`; this view exists so that decision
/// has a single place to grow (e.g. when a `transferring` phase is
/// added for spec §10 mid-deployment transfer).
struct OnboardingFlow: View {

    @Environment(OnboardingState.self) private var onboarding

    var body: some View {
        switch onboarding.phase {
        case .adminProvisioning:
            AdminWizardView()
        case .microscopistClaim:
            MicroscopistWizardView()
        case .complete:
            // Belt-and-braces: the composition root should never mount
            // OnboardingFlow when phase is `.complete`, but if a future
            // refactor regresses that gating an empty view keeps the
            // app navigable rather than crashing.
            EmptyView()
        }
    }
}
