import SwiftUI
import UIKit

/// Data-management screen. Per spec §11 the three affordances are
/// Export (Phase 13 — wired here), Lock device (live), and Reset device
/// (Phase 11 — wired since the previous phase).
struct DataManagementView: View {

    @Environment(AuthGate.self) private var authGate
    @Environment(ExportService.self) private var exportService

    @State private var showLockConfirmation = false
    @State private var isExporting = false
    @State private var exportError: String?
    @State private var shareURL: URL?

    var body: some View {
        List {
            Section {
                Button {
                    Task { await runExport() }
                } label: {
                    HStack {
                        Label("Export all data", systemImage: "square.and.arrow.up")
                        Spacer()
                        if isExporting {
                            ProgressView()
                        }
                    }
                }
                .disabled(isExporting)
            } footer: {
                Text("Export creates a portable, signed ZIP bundle of every prediction, audit entry, clinician profile, and consent record on this device. Share via AirDrop, Files, or Mail.")
            }

            Section {
                Button(role: .destructive) {
                    showLockConfirmation = true
                } label: {
                    Label("Lock device", systemImage: "lock.fill")
                }
            } footer: {
                Text("Locks the current session immediately. Biometric or device passcode required to unlock.")
            }

            Section {
                NavigationLink {
                    ResetDeviceView()
                } label: {
                    Label("Reset device", systemImage: "trash")
                        .foregroundStyle(.red)
                }
            } footer: {
                Text("Wipes the clinician profile and returns the device to admin provisioning. Predictions and audit history are preserved as chain-of-custody.")
            }
        }
        .navigationTitle("Data management")
        .alert("Lock device?", isPresented: $showLockConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Lock", role: .destructive) {
                authGate.lockSession(reason: .manual)
            }
        } message: {
            Text("The next person using this device must re-authenticate with biometrics or the device passcode.")
        }
        .alert("Export failed", isPresented: Binding(
            get: { exportError != nil },
            set: { if !$0 { exportError = nil } }
        )) {
            Button("OK", role: .cancel) { exportError = nil }
        } message: {
            Text(exportError ?? "")
        }
        .sheet(item: Binding(
            get: { shareURL.map { ShareItem(url: $0) } },
            set: { shareURL = $0?.url }
        )) { item in
            ShareSheet(url: item.url)
        }
    }

    @MainActor
    private func runExport() async {
        isExporting = true
        defer { isExporting = false }
        do {
            // Spec §9: Export all data is a fresh-auth action. Prompt
            // biometrics regardless of current session state.
            try await authGate.unlock(reason: "Authenticate to export data")
            let url = try await exportService.generateBundle()
            shareURL = url
        } catch {
            exportError = error.localizedDescription
        }
    }
}

/// Identifiable wrapper so `.sheet(item:)` can re-present after the user
/// dismisses the share sheet and a fresh export is initiated.
private struct ShareItem: Identifiable {
    let url: URL
    var id: URL { url }
}

/// Thin `UIViewControllerRepresentable` around `UIActivityViewController`
/// for the share-sheet hand-off per spec §14.
private struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
