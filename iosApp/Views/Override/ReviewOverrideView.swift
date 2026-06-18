import SwiftData
import SwiftUI
@preconcurrency import Shared

/// Single-screen review-override form (spec §12 review override).
///
/// Surfaces all required fields at once: corrected verdict, reason from the
/// five canonical `Shared.OverrideReason` cases, override-by initials
/// defaulting to the device clinician's profile but overwritable per spec
/// §12 second-reviewer workaround, optional notes, and the mandatory
/// "I have reviewed the full session context" checkbox.
///
/// Save is gated on verdict + reason + checkbox being set (state machine
/// asserted in `ReviewOverrideTests`). Tapping save triggers a fresh
/// biometric prompt via `AuthGate.unlock(...)` (spec §9 lists review
/// override as a fresh-auth action) and only on success does it call
/// `PredictionStore.override(...)`, which writes the prediction columns
/// plus an `override_recorded` audit entry with `contextReviewed = true`.
///
/// Live override (the 2-tap modal during active screening) is a separate
/// surface; it lands in Phase 8 with the camera UI.
struct ReviewOverrideView: View {

    let prediction: Prediction

    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext
    @Environment(PredictionStore.self) private var predictionStore
    @Environment(AuthGate.self) private var authGate

    enum Verdict: String, CaseIterable, Identifiable {
        case parasitized = "Parasitized"
        case uninfected = "Uninfected"

        var id: String { rawValue }
    }

    @State private var verdict: Verdict? = nil
    @State private var reason: Shared.OverrideReason? = nil
    @State private var initials: String = ""
    @State private var notes: String = ""
    @State private var contextReviewed: Bool = false

    @State private var isSaving: Bool = false
    @State private var errorText: String? = nil
    @State private var didLoadDefaults: Bool = false

    private var probabilityPercent: Int {
        Int((prediction.parasitizedProb * 100).rounded())
    }

    /// Save-enable state machine per spec §12: verdict picked, reason
    /// chosen, and the context-reviewed checkbox ticked. Initials and
    /// notes are not required. Pulled out as a `static` so the unit
    /// tests can assert the rule without spinning up a SwiftUI view.
    static func canSave(
        verdict: Verdict?,
        reason: Shared.OverrideReason?,
        contextReviewed: Bool
    ) -> Bool {
        verdict != nil && reason != nil && contextReviewed
    }

    private var canSave: Bool {
        Self.canSave(
            verdict: verdict,
            reason: reason,
            contextReviewed: contextReviewed
        )
    }

    var body: some View {
        Form {
            headerSection
            verdictSection
            reasonSection
            actorSection
            notesSection
            checkboxSection
            if let errorText {
                Section {
                    Text(errorText)
                        .foregroundStyle(.red)
                }
            }
        }
        .navigationTitle("Review and override")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel") { dismiss() }
                    .disabled(isSaving)
            }
            ToolbarItem(placement: .confirmationAction) {
                Button("Save override") {
                    Task { await save() }
                }
                .disabled(!canSave || isSaving)
            }
        }
        .onAppear {
            loadDefaultsIfNeeded()
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 4) {
                Text("The model said: \(prediction.label) (\(probabilityPercent)%)")
                    .font(.headline)
                Text("Captured: \(prediction.timestamp.iso8601UtcLong), session \(String(prediction.sessionId.prefix(8)))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var verdictSection: some View {
        Section("Corrected verdict") {
            Picker("Corrected verdict", selection: $verdict) {
                Text("Choose…").tag(Verdict?.none)
                ForEach(Verdict.allCases) { v in
                    Text(v.rawValue).tag(Verdict?.some(v))
                }
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Corrected verdict")
        }
    }

    private var reasonSection: some View {
        Section("Reason") {
            Picker("Reason", selection: $reason) {
                Text("Choose…").tag(Shared.OverrideReason?.none)
                ForEach(Shared.OverrideReason.entries, id: \.canonical) { r in
                    Text(Self.displayLabel(for: r))
                        .tag(Shared.OverrideReason?.some(r))
                }
            }
            .pickerStyle(.menu)
        }
    }

    private var actorSection: some View {
        Section("Override by") {
            TextField("Initials", text: $initials)
                .textInputAutocapitalization(.characters)
                .autocorrectionDisabled(true)
                .onChange(of: initials) { _, newValue in
                    if newValue.count > 2 {
                        initials = String(newValue.prefix(2))
                    }
                }
            Text("Defaults to the device clinician's initials. Overwrite if a second reviewer recorded this override.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var notesSection: some View {
        Section("Notes") {
            TextField(
                "Optional notes",
                text: $notes,
                axis: .vertical
            )
            .lineLimit(5, reservesSpace: true)
            .autocorrectionDisabled(false)
        }
    }

    private var checkboxSection: some View {
        Section {
            Toggle(isOn: $contextReviewed) {
                Text("I have reviewed the full session context for this prediction")
            }
            .accessibilityIdentifier("contextReviewedToggle")
        } footer: {
            Text("Required to enable Save.")
                .font(.caption)
        }
    }

    // MARK: - Save flow

    private func save() async {
        guard canSave else { return }
        guard let verdict, let reason else { return }
        errorText = nil
        isSaving = true
        defer { isSaving = false }

        do {
            try await authGate.unlock(reason: "Confirm review override")
        } catch {
            errorText = error.localizedDescription
            return
        }

        do {
            try predictionStore.override(
                prediction,
                verdict: verdict.rawValue,
                context: Shared.OverrideContext.review.canonical,
                reason: reason.canonical,
                notes: notes.isEmpty ? nil : notes,
                actorInitials: initials.isEmpty ? nil : initials,
                contextReviewed: true
            )
            dismiss()
        } catch {
            errorText = error.localizedDescription
        }
    }

    // MARK: - Defaults

    private func loadDefaultsIfNeeded() {
        guard !didLoadDefaults else { return }
        didLoadDefaults = true
        if let profile = try? ClinicianRepository(context: modelContext).current(),
           let storedInitials = profile.initials, !storedInitials.isEmpty {
            initials = storedInitials
        }
    }

    // MARK: - Display labels

    /// Spec §15: override-reason labels stay English in the UI even when
    /// the rest of the app is localised. The mapping is centralised here
    /// so the canonical enum stays the single source of truth.
    static func displayLabel(for reason: Shared.OverrideReason) -> String {
        switch reason {
        case .imageQuality:
            return "Image quality"
        case .atypicalMorphology:
            return "Atypical morphology"
        case .modelFalsePositive:
            return "Model false-positive"
        case .modelFalseNegative:
            return "Model false-negative"
        case .other:
            return "Other"
        default:
            return reason.canonical
        }
    }
}

private extension Date {
    /// ISO 8601 UTC down to the second, e.g. "2026-03-14 09:23:45 UTC".
    /// Matches spec §12 review-override mock header format.
    var iso8601UtcLong: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm 'UTC'"
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: self)
    }
}
