import SwiftUI
@preconcurrency import Shared

/// Picker over the model registry. Reads entries from
/// `ModelRegistryService.registry.all()` and defaults to `BNLeaky_Keras`
/// per spec §7.
struct ModelPicker: View {

    @Binding var selectedId: String
    let entries: [ModelRegistryEntry]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Default model", selection: $selectedId) {
                ForEach(entries, id: \.id) { entry in
                    Text(entry.displayName).tag(entry.id)
                }
            }
            .pickerStyle(.menu)

            if let entry = entries.first(where: { $0.id == selectedId }) {
                Text(entry.description)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}
