import SwiftUI
@preconcurrency import Shared

/// Picker over the three `Shared.LawfulBasis` enum cases, with the
/// one-line explanations mandated by spec §10 step 5. Like `Jurisdiction`,
/// these display labels stay in English (spec §15).
struct LawfulBasisPicker: View {

    @Binding var selection: LawfulBasis

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Lawful basis", selection: $selection) {
                ForEach(Self.options, id: \.0) { option in
                    Text(option.1).tag(option.0)
                }
            }
            .pickerStyle(.menu)

            Text(Self.explanation(for: selection))
                .font(.footnote)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    static let options: [(LawfulBasis, String)] = [
        (.explicitConsent, "Explicit consent"),
        (.vitalInterests, "Vital interests"),
        (.healthProvision, "Provision of health care"),
    ]

    /// Per spec §10 step 5: "Lawful basis picker … with one-line
    /// explanations." Kept short enough to fit a single line on a
    /// portrait iPhone without truncation.
    static func explanation(for basis: LawfulBasis) -> String {
        switch basis {
        case .explicitConsent:
            "Patient has given specific, informed consent for screening."
        case .vitalInterests:
            "Screening is necessary to protect the life of the patient."
        case .healthProvision:
            "Screening is part of medical care this clinic provides."
        default:
            ""
        }
    }
}
