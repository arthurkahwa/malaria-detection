import SwiftUI

/// Phase 2 sub-coordinator. Dispatches on
/// `OnboardingState.microscopistStep` (spec §10).
struct MicroscopistWizardView: View {

    @Environment(OnboardingState.self) private var onboarding

    var body: some View {
        switch onboarding.microscopistStep {
        case .welcome:
            MicroscopistWelcomeStep()
        case .initials:
            InitialsStep()
        case .biometric:
            MicroscopistBiometricStep()
        case .orientation:
            OrientationStep()
        }
    }
}
