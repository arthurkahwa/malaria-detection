import SwiftUI

/// Settings tab. Per spec §11 the tab is always accessible (no auth gate),
/// but every editable row triggers a fresh biometric prompt via the
/// individual edit screens (spec §9 partial-lock model).
///
/// Sections render in spec-prescribed order:
///   1. Clinic (read-only)
///   2. Clinician profile
///   3. Inference
///   4. Models
///   5. Language
///   6. Legal
///   7. Crash logs (Phase 14 — disabled)
struct SettingsTab: View {
    var body: some View {
        NavigationStack {
            Form {
                ClinicSection()
                ClinicianSection()
                InferenceSection()
                ModelsSection()
                LanguageSection()
                LegalSection()
                CrashLogsSection()
            }
            .navigationTitle("Settings")
        }
    }
}
