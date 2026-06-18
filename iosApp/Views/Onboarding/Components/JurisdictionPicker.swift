import SwiftUI
@preconcurrency import Shared

/// Picker over the six `Shared.Jurisdiction` enum cases. Display strings
/// stay in English (spec §15: jurisdiction labels are English-only
/// because clinic admins routinely cross-reference them with regulatory
/// documents that are themselves English).
///
/// The persisted value is the `canonical` lowercase_snake form so it
/// matches what the audit log and shared `RetentionPolicy.minimumYears`
/// already expect.
struct JurisdictionPicker: View {

    @Binding var selection: Jurisdiction

    var body: some View {
        Picker("Jurisdiction", selection: $selection) {
            ForEach(Self.options, id: \.0) { option in
                Text(option.1).tag(option.0)
            }
        }
        .pickerStyle(.menu)
    }

    /// `(case, displayString)` pairs in the order the spec presents them.
    static let options: [(Jurisdiction, String)] = [
        (.usHipaa, "United States — HIPAA"),
        (.euGdprDe, "EU GDPR (Germany)"),
        (.euGdprFr, "EU GDPR (France)"),
        (.euGdprGeneric, "EU GDPR (other member state)"),
        (.keDpa, "Kenya — Data Protection Act"),
        (.other, "Other / not listed"),
    ]
}
