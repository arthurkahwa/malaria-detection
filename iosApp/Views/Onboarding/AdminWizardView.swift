import SwiftUI

/// Phase 1 sub-coordinator. Dispatches on `OnboardingState.adminStep`
/// (spec §10) and renders the matching step view. Step views drive
/// transitions by calling `OnboardingState.*` methods.
struct AdminWizardView: View {

    @Environment(OnboardingState.self) private var onboarding

    var body: some View {
        switch onboarding.adminStep {
        case .language:
            LanguageStep()
        case .welcome:
            WelcomeStep()
        case .licenseAck:
            LicenseAckStep()
        case .disclaimerAck:
            DisclaimerAckStep()
        case .clinicDetails:
            ClinicDetailsStep()
        case .inferencePolicy:
            InferencePolicyStep()
        case .biometric:
            AdminBiometricStep()
        case .provisioningComplete:
            ProvisioningCompleteStep()
        }
    }
}
