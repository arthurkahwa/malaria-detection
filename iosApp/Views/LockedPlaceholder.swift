import SwiftUI

/// Shown on Home and History tabs when `AuthGate.state == .locked`.
/// Tapping triggers biometric unlock once Phase 6 lands.
struct LockedPlaceholder: View {
    @Environment(AuthGate.self) private var authGate

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Spacer()
            HStack {
                Spacer()
                VStack(alignment: .leading, spacing: 12) {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 56))
                    Text("Tap to unlock")
                        .font(.title2)
                }
                Spacer()
            }
            Spacer()
        }
        .padding(.leading, 24)
        .padding(.trailing, 24)
        .contentShape(Rectangle())
        .onTapGesture {
            Task { try? await authGate.unlock() }
        }
    }
}
