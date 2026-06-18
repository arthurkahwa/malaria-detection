import SwiftUI

/// A checkbox bound to a Bool. Used on the Hippocratic License and
/// medical-device disclaimer steps where the spec requires an explicit
/// "I have read and accept" acknowledgement before Continue activates.
struct ConsentCheckbox: View {

    @Binding var isChecked: Bool
    let label: String

    var body: some View {
        Button {
            isChecked.toggle()
        } label: {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: isChecked ? "checkmark.square.fill" : "square")
                    .foregroundStyle(isChecked ? Color.accentColor : Color.secondary)
                    .font(.title3)
                Text(label)
                    .multilineTextAlignment(.leading)
                    .foregroundStyle(.primary)
                Spacer(minLength: 0)
            }
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isChecked ? [.isSelected] : [])
    }
}
