import SwiftUI

/// Consistent layout for every onboarding wizard step (spec §10).
///
/// All steps share the same skeleton: title + scrollable body + primary
/// action button pinned to the bottom, with an optional secondary action
/// rendered above the primary one.
///
/// RTL-readiness (spec §11): uses `.leading` / `.trailing` padding and
/// alignment exclusively. Never `.left` / `.right`.
struct WizardStepContainer<Body: View>: View {

    let title: String
    let subtitle: String?
    let primaryAction: PrimaryAction
    let secondaryAction: SecondaryAction?
    let stepIndicator: String?
    @ViewBuilder var content: () -> Body

    init(
        title: String,
        subtitle: String? = nil,
        stepIndicator: String? = nil,
        primary: PrimaryAction,
        secondary: SecondaryAction? = nil,
        @ViewBuilder content: @escaping () -> Body
    ) {
        self.title = title
        self.subtitle = subtitle
        self.stepIndicator = stepIndicator
        self.primaryAction = primary
        self.secondaryAction = secondary
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let stepIndicator {
                Text(stepIndicator)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 24)
                    .padding(.leading, 24)
                    .padding(.trailing, 24)
            }
            Text(title)
                .font(.largeTitle.weight(.semibold))
                .padding(.top, stepIndicator == nil ? 24 : 8)
                .padding(.leading, 24)
                .padding(.trailing, 24)
            if let subtitle {
                Text(subtitle)
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .padding(.top, 8)
                    .padding(.leading, 24)
                    .padding(.trailing, 24)
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    content()
                }
                .padding(.top, 24)
                .padding(.leading, 24)
                .padding(.trailing, 24)
                .padding(.bottom, 16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            VStack(spacing: 8) {
                if let secondaryAction {
                    Button(action: secondaryAction.handler) {
                        Text(secondaryAction.title)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                }
                Button(action: primaryAction.handler) {
                    Text(primaryAction.title)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!primaryAction.isEnabled)
            }
            .padding(.leading, 24)
            .padding(.trailing, 24)
            .padding(.bottom, 32)
            .padding(.top, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }
}

extension WizardStepContainer {
    struct PrimaryAction {
        let title: String
        let isEnabled: Bool
        let handler: () -> Void

        init(title: String, isEnabled: Bool = true, handler: @escaping () -> Void) {
            self.title = title
            self.isEnabled = isEnabled
            self.handler = handler
        }
    }

    struct SecondaryAction {
        let title: String
        let handler: () -> Void
    }
}
