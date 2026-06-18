# MANUAL_TEST_PLAN

> **Status:** first-pass for v0.1. Source:
> [`KMP_App_Specification.md`](../KMP_App_Specification.md) §20 ("Manual
> test plan") and §22 (Phase 16 Documentation deliverable).

This document covers what the autonomous test suite cannot. Each section
is run on both platforms before a v0.x release, on at least one real
iPhone and one real Android device.

## What the autonomous suite covers (and does not)

The autonomous suite runs in CI on every PR:

- Shared module — `Threshold`, `SessionGrouping`, `Permissions`,
  `RetentionPolicy`, `ModelRegistry`, `Preprocessor`, the domain enums,
  and the Phase 13 export package + Phase 14 crash log shared types
- iOS — SwiftData CRUD, classifier bridge, onboarding state machine,
  Phase 6 / 9 / 10 / 11 / 13 / 14 unit tests on iPhone 17 Pro simulator
  under iOS 26.0 (`63 / 63` last verified)
- Android — Room schema drift, onboarding state machine, settings store,
  session stats, review override state, crash log store, export
  service — JVM unit tests (`45 / 45` last verified)
- Schema drift — `SchemaDriftTest` (JVM, Android) and the iOS SwiftData
  reflection equivalent run on every PR

The autonomous suite **does not** cover:

- Real biometric hardware (`LAContext` / `BiometricPrompt`) — simulators
  cannot exercise the actual TEE / Secure Enclave paths
- Real camera frames — the iOS Simulator does not vend `CMSampleBuffer`s;
  Android emulators vary
- Real network behaviour for Hugging Face model downloads (the
  downloader is not yet implemented; this is a placeholder)
- Auto-logout via `SessionTimer` (not yet wired on either platform — see
  *Known gaps* below)
- RTL layout integrity under the platform's "Force RTL" developer
  option — required before v0.1 ships per spec §11
- Onboarding flow on a *fresh* device (the on-device DB is wiped between
  CI runs but composition-root behaviour is best verified by hand)
- Permission-denied flows when the OS-level setting is actually off
- ScenePhase backgrounding behaviour on a real device

## Per-flow platform and runtime matrix

| Flow | iOS Simulator | iOS Device | Android Emulator | Android Device |
|---|---|---|---|---|
| 1. Bootstrap | Yes | Yes | Yes | Yes |
| 2. Onboarding Phase 1 (admin) | Partial (biometric is faked via Hardware → Face ID → "Matching Face") | Yes | Partial (biometric via `adb -e emu finger touch 1`) | Yes |
| 3. Onboarding Phase 2 (microscopist) | Partial (biometric faked) | Yes | Partial (biometric via adb) | Yes |
| 4. Active screening | **No — Simulator does not vend camera frames** | **Yes (required)** | **N/A — Phase 3 blocked** | **N/A — Phase 3 blocked** |
| 5. History viewer | Yes | Yes | Yes | Yes |
| 6. Review override | Partial (biometric faked) | Yes | Partial (biometric via adb) | Yes |
| 7. Settings | Partial (biometric faked) | Yes | Partial (biometric via adb) | Yes |
| 8. Export | Partial (share sheet renders; recipient apps limited on Simulator) | Yes | Partial (share sheet renders to limited targets) | Yes |
| 9. Crash logs | Yes (force a crash via the iOS Simulator's debug menu — see *Known gaps*) | Yes | Yes (force-stop + ANR) | Yes |
| 10. RTL verification | Yes (Xcode scheme option) | Yes | Yes (Developer options → Force RTL) | Yes |
| 11. Lock + auto-logout | Partial — see *Known gaps*: `SessionTimer` not wired | Partial | Partial | Partial |

**Biometric simulation shortcuts.**
*iOS Simulator:* Hardware → Face ID → Enrolled, then Hardware → Face ID →
Matching Face / Non-matching Face during a prompt. Touch ID equivalent
under the same menu on simulators provisioned for older device classes.
*Android Emulator:* `adb -e emu finger touch <id>` to inject a fingerprint
event during a prompt. Face unlock is not faked by the emulator; tests
involving face unlock require a real device.

**Tester reporting convention.** Tests are numbered `<flow>.<item>` (e.g.,
5.3, 9.2). A tester reports "passed 1.1, 1.2, 1.3, failed 5.3, untested
7.4" without ambiguity.

---

## 1. Bootstrap (fresh-install behaviour)

Install the app onto a device that has never run it. No prior data.

- [ ] **1.1** iOS — App launches; first screen is the language picker (no
      `LAContext` prompt yet, no Home tab visible)
- [ ] **1.2** Android — App launches; first screen is the language picker
      (no `BiometricPrompt` yet, no Home tab visible)
- [ ] **1.3** Both — On launch, no audit entries exist for the previous
      install (a fresh install starts with an empty audit log; reinstall
      means a clean device from the app's point of view)
- [ ] **1.4** Both — The medical-device disclaimer, license
      acknowledgement, and onboarding-complete state are all in their
      pre-onboarded state (no consent records persisted)

Acceptance: The app comes up to language picker, with no Home / History /
Settings tabs visible, and no `auth_success` audit entries from a prior
session.

---

## 2. Onboarding Phase 1 (admin)

Run end-to-end immediately after a fresh install. Each step should result
in the listed audit entry and / or consent record.

- [ ] **2.1** Language picker — selecting a language persists across the
      reset-device flow (spec §15). Acceptance: language survives reset.
- [ ] **2.2** Welcome screen — Continue button is enabled, no gating
- [ ] **2.3** Hippocratic License acknowledgement — Continue is **gated**
      on the "I have read and accept" checkbox. Audit:
      `clinic_configured` is **not** yet written (it lands at 2.6); a
      `ConsentRecord` with `consentType = hippocratic_license` is
      written on tap-through
- [ ] **2.4** Medical-device disclaimer — Continue is **gated** on the "I
      understand this is decision support" checkbox. A `ConsentRecord`
      with `consentType = medical_disclaimer` is written on tap-through.
      The spec §16 crash-log disclosure line is present
- [ ] **2.5** Clinic details form — Clinic name (required), jurisdiction
      picker (6 canonical values), lawful basis picker (3 canonical
      values, each with a one-line explanation). Form validation: empty
      clinic name disables Continue
- [ ] **2.6** Inference policy — default model is the bundled
      `Malaria_BNLeaky_Keras`; the 17 other registry entries are listed
      with a "Requires download (not in v1)" caption (see *Known gaps*).
      Threshold slider 0.10–0.90 default 0.30. Continue writes
      `clinic_configured` and `inference_policy_set` audit entries with
      the selected values
- [ ] **2.7** Admin biometric enrolment — biometric prompt appears, real
      device matches succeed and write `admin_biometric_enrolled` audit
      entry. **Failure path:** twice-failed biometric falls back to
      passcode / device credential
- [ ] **2.8** No-biometric / no-passcode device — app refuses to enrol
      and deep-links to system settings to configure biometric or
      passcode. No `admin_biometric_enrolled` entry written until
      configured
- [ ] **2.9** "Device provisioned for [Clinic]" interstitial — the
      Clinic name from step 2.5 appears verbatim. Continue advances to
      microscopist welcome
- [ ] **2.10** All Phase 1 audit entries are present in order:
      `admin_provisioning_started` → `clinic_configured` →
      `inference_policy_set` → `admin_biometric_enrolled` →
      `admin_provisioning_completed`

Acceptance: Phase 1 completes end-to-end; restarting the app at any
intermediate step resumes at the same step (composition root reads
`OnboardingState.phase`).

---

## 3. Onboarding Phase 2 (microscopist)

Continues directly from the Phase 1 handoff interstitial, or — for a
single-deployer flow — runs back-to-back.

- [ ] **3.1** Welcome — Continue is enabled, no gating
- [ ] **3.2** Initials field — accepts 0–2 ASCII characters; rejects
      emoji, control characters, and any character outside the
      basic-Latin range. Empty initials are valid (optional field)
- [ ] **3.3** Microscopist biometric enrolment — biometric prompt
      appears; success writes `microscopist_biometric_enrolled` audit
      entry
- [ ] **3.4** 3-page orientation walkthrough (Capture → Override →
      History). "Begin screening" CTA on page 3 calls
      `OnboardingState.finishOrientation()`, which flips
      `phase = complete`
- [ ] **3.5** Composition root re-renders with the operational tab shell
      (Home, History, Settings) once `phase = complete`
- [ ] **3.6** All Phase 2 audit entries are present in order:
      `microscopist_claim_started` → `microscopist_biometric_enrolled`
      → `microscopist_claim_completed`
- [ ] **3.7** Orientation: rotate the device through portrait and the
      two landscape orientations — orientation is locked to portrait per
      spec §11. Acceptance: device rotation does not rotate UI

Acceptance: Phase 2 completes end-to-end; the user lands on Home;
Settings → Clinician profile shows the just-set initials.

---

## 4. Active screening (iOS only — Android blocked on Phase 3)

**Requires a real iPhone — the Simulator does not vend `CMSampleBuffer`s,
so `CameraService.captureOneFrame()` throws `captureTimeout`.**

Android live screening is blocked on Phase 3 (TFLite export pipeline);
this section is iOS-only until then.

- [ ] **4.1** First-time camera permission prompt appears at the right
      moment (on entering Home tab, not at app launch). iOS:
      `NSCameraUsageDescription` renders correctly
- [ ] **4.2** Permission-denied path — a fallback view explains the
      denial and deep-links to `UIApplication.openSettingsURLString`
- [ ] **4.3** Permission-granted path — `AVCaptureVideoPreviewLayer`
      renders the back-camera feed in `.resizeAspectFill`, portrait
      orientation, occupying the centre of the screen
- [ ] **4.4** Top: model badge displays the active model name
      (`Malaria_BNLeaky_Keras`)
- [ ] **4.5** Bottom: Capture button is enabled when a frame is available
- [ ] **4.6** Tap Capture — within ~200 ms, an inline prediction overlay
      appears with: verdict label (Parasitized / Uninfected), confidence
      percentage, `RiskBandIndicator` for the risk band (low / gray zone
      / high), Override button, End session button
- [ ] **4.7** A `Prediction` row is persisted with `inferenceMs` populated;
      a `prediction_created` audit entry is written
- [ ] **4.8** Live override 2-tap flow — Override → verdict picker →
      reason picker → dismiss. **No biometric, no notes, no initials**
      per spec §12. A single `override_recorded` audit entry with
      `overrideContext = "live"`, `contextReviewed = nil` is written
- [ ] **4.9** End session — `cameraService.stop()` is called; UI returns
      to the idle Home state. No additional audit entry per spec §11
- [ ] **4.10** ScenePhase backgrounding — sending the app to background
      stops the capture session; resuming foregrounds the idle Home
- [ ] **4.11** `AuthGate` lock (manual lock from Settings) stops the
      capture session immediately
- [ ] **4.12** Permission revoked while app is open — gracefully detected
      on resume; the permission-denied fallback view appears

Acceptance: Capture → prediction → override → end session round-trip
works on real iPhone; all audit entries land correctly; no `Prediction`
row holds an image.

---

## 5. History viewer

The History tab is auth-gated. Each of the five subsections has its own
checks.

- [ ] **5.1** Recent predictions — predictions are sorted by `capturedAt`
      desc; only the persistence layer's read surface is exercised
      (no inference)
- [ ] **5.2** Flagged for review — predicate is `flagged == true AND
      override == nil`. Acceptance: a flagged prediction that's been
      overridden does **not** appear; a gray-zone prediction with no
      override does
- [ ] **5.3** Sessions — grouped by `sessionId` via the 30-minute gap
      rule. Stats: count, first capture, last capture, parasitized count
- [ ] **5.4** Audit log — action picker (canonical strings) and date
      range; capped at 200 most-recent entries
- [ ] **5.5** Data management — Export all data, Reset device (see flows
      7 and 8 below)
- [ ] **5.6** Tap a prediction → AI Analysis detail view. A
      `prediction_viewed` audit entry is written **once per mount** (guarded
      by `@State didAudit` on iOS, `LaunchedEffect(predictionId)` on
      Android keyed to prediction id). Acceptance: navigate away and
      back — no second `prediction_viewed` audit row appears
- [ ] **5.7** Tap a session → Session detail with header stats. The
      relabel field rejects non-ASCII characters (emoji, em-dash,
      accents, CJK) per spec §13. Empty after trimming is rejected.
      Acceptance: only valid ASCII relabels write a `session_relabeled`
      audit row
- [ ] **5.8** Session detail: "Mark as duplicate" — pick a target from
      the last 50 predictions in the same session; writes a
      `prediction_linked_as_duplicate` audit row with the target id in
      metadata
- [ ] **5.9** Audit Entry detail — `metadataJson` parses and pretty-
      prints; no raw escapes visible
- [ ] **5.10** Personal-data warning is surfaced on the session relabel
      field per spec §13 ("relabel persists across export — do not enter
      patient names")

Acceptance: Each subsection works end-to-end; the `prediction_viewed`
once-per-mount semantics hold on both platforms.

---

## 6. Review override

Triggered from a flagged prediction in History. Single-screen form per
spec §12.

- [ ] **6.1** From a flagged prediction without an existing override, the
      "Review and override" affordance is visible
- [ ] **6.2** From a prediction with `clinicianOverride != nil`, the
      "Review and override" affordance is **hidden** per spec §12
      ("override cannot be undone in v1")
- [ ] **6.3** Form header: "The model said: <label> (<%>)" + capture
      timestamp + session prefix
- [ ] **6.4** Corrected verdict — segmented picker on iOS / FilterChip
      row on Android, two options (Parasitized / Uninfected)
- [ ] **6.5** Reason picker — DropdownMenu / Picker, 5 canonical options
      with English-only labels per spec §15
- [ ] **6.6** Override-by initials — defaults to the device clinician's
      initials, capped at 2 chars
- [ ] **6.7** Notes — optional, multi-line (3–5 lines)
- [ ] **6.8** Required checkbox: "I have reviewed the full session
      context for this prediction"
- [ ] **6.9** Save button is disabled until verdict + reason + checkbox
      are all set; enabled when all three are
- [ ] **6.10** On Save — a **fresh** biometric prompt appears
      (`AuthGate.unlock(reason: "Confirm review override")` on iOS,
      `BiometricPrompter.prompt(...)` on Android). On success,
      `PredictionStore.override(...)` writes the override columns
- [ ] **6.11** Exactly **one** `override_recorded` audit entry is written
      with `overrideContext = "review"`, `overrideReason = <canonical>`,
      `overrideActorInitials`, `contextReviewed = true`, optional
      `overrideNotes`. Acceptance: count strictly +1 after the save
- [ ] **6.12** Biometric failure path — Save is cancelled, no override
      columns or audit entry written, user stays on the form

Acceptance: A flagged prediction can be overridden in review context;
chain-of-custody preserved; second override on the same prediction is
not possible from the UI.

---

## 7. Settings

Each editable row gates on a fresh biometric prompt (spec §9).

- [ ] **7.1** Clinic — read-only, shows clinic name + jurisdiction +
      lawful basis
- [ ] **7.2** Clinician profile — actorId UUID copyable; role read-only;
      initials editable. Edit triggers biometric → writes `profile_updated`
      audit entry with `field=initials` metadata
- [ ] **7.3** Inference — for admin role, threshold + default model +
      auto-logout editable; for microscopist role, all are read-only per
      spec §11. Each edit triggers biometric and writes a matching audit
      entry (`threshold_changed`, `default_model_changed`,
      `auto_logout_changed`) with `old_value` + `new_value` metadata
- [ ] **7.4** Threshold — slider 0.10–0.90 with 0.05 steps. Acceptance:
      out-of-range values are rejected; the new threshold is reflected
      on the next capture
- [ ] **7.5** Default model — Bundled / Downloaded / Available
      subsections. Only the bundled `Malaria_BNLeaky_Keras` is selectable
      in v1; the 17 remote entries show "Requires download (not in v1)"
- [ ] **7.6** Auto-logout — 5 / 15 / 30 minute picker
- [ ] **7.7** Language — biometric-gated edit per spec §11 ("to prevent
      stranger-flips"). Selection writes `language_changed` audit entry;
      app reloads to apply the new locale
- [ ] **7.8** Legal — links to Privacy policy, Terms of service,
      Decision-support disclaimer (`LegalText.notice`), and open-source
      acknowledgements
- [ ] **7.9** Crash logs (see flow 9 below)
- [ ] **7.10** Reset device flow:
      - Settings or History → Data management → Reset device requires
        admin biometric
      - Double-confirmation dialog with PII warning
      - Reset wipes the clinician row; preserves predictions and audit
        history; writes a `device_reprovisioned` audit entry with
        `metadata.wiped_actor_id` set
      - Device returns to `provisioned-unclaimed`; composition root
        auto-mounts `OnboardingFlow`
      - Phase 1 provisioning audit entries from prior installs are
        preserved (chain-of-custody)
      - Clinic-level config (jurisdiction, threshold, default model)
        is NOT preserved by `ClinicianRepository.wipe()` in v1; on
        re-onboarding the admin re-enters them. Settings shows the
        prior values until re-onboarding completes since
        `SettingsStore` re-hydrates from the audit log on `Reset`

Acceptance: All editable rows are biometric-gated and write the expected
audit entries; reset-device preserves chain-of-custody.

---

## 8. Export bundle

From History → Data management → Export all data. Spec §14.

- [ ] **8.1** Tap Export — fresh biometric prompt appears; success
      triggers `ExportService.generateBundle()`
- [ ] **8.2** `export_initiated` audit entry written on tap (before the
      bundle generation)
- [ ] **8.3** Bundle generation completes; share sheet appears
      (`UIActivityViewController` on iOS, `Intent.ACTION_SEND` on
      Android)
- [ ] **8.4** `export_completed` audit entry written with `size` +
      `signature` metadata
- [ ] **8.5** Filename pattern matches spec §14 (timestamped)
- [ ] **8.6** Failure path — missing clinic config writes `export_failed`
      with `reason` metadata. Acceptance: failure surfaces a user-
      facing error and the audit row lands
- [ ] **8.7** Share the bundle to a known target (e.g., Files / Email /
      AirDrop). Open `export.json` in a text editor and confirm:
      - `schemaVersion = "1.0"`
      - `signature` field present
      - All timestamps are ISO-8601 UTC
      - JSON keys are sorted (canonical form)
- [ ] **8.8** Verify the signature off-device using the documented HMAC
      key derivation (`SHA-256(deviceUuid + ":" + exportTimestamp)`).
      Acceptance: signature verifies over the unsigned form of the
      payload
- [ ] **8.9** Byte-identical-between-platforms — exports from iOS and
      Android of the same content produce the same bytes (CI covers
      this; manual sanity-check via diff on a small dataset)

Acceptance: Export round-trip works on both platforms; bundle is signed
and parseable off-device.

---

## 9. Crash logs

Spec §16. Each on-device crash should produce one `{incident-uuid}.json`.

**Known gap:** there is no in-app debug menu to force a crash. To
exercise the writer end-to-end, attach Xcode / Android Studio in debug
and trigger an `NSException` / `RuntimeException` manually, or wire a
temporary "crash" button into a debug build.

- [ ] **9.1** Install the app fresh; confirm `~/Documents/crashlogs/`
      (iOS) / `context.filesDir/crashlogs/` (Android) is empty
- [ ] **9.2** Force a crash. On next launch, the crash handler ran
      during the prior launch's tear-down; a `{incident-uuid}.json` is
      now on disk
- [ ] **9.3** Open Settings → Crash logs. The new log appears with the
      incident UUID, timestamp, app/OS version, and device model class
- [ ] **9.4** The log contents include: stack trace, last 50 audit
      action canonical strings (via `RecentActionRing`), memory readout,
      unlocked/locked flag. The log contents **exclude** prediction
      data, override notes, clinician initials, actor UUIDs, image
      hashes, clinic config, consent records (spec §16 forbidden fields)
- [ ] **9.5** Tap the log → platform share sheet. Sharing writes a
      `crash_log_shared` audit entry with the incident UUID and empty
      metadata
- [ ] **9.6** 30-day expiry sweep — backdate a log file's mtime to 31
      days ago (via Files on iOS / `adb shell` on Android), relaunch the
      app, confirm the file is gone
- [ ] **9.7** Survive a reset-device — perform Reset device, then return
      to Settings → Crash logs. Pre-reset crash logs are still listed
      (filesystem, not in SwiftData / Room)
- [ ] **9.8** Android — confirm the file is encrypted via
      `androidx.security.crypto.EncryptedFile` (raw read with no key
      fails to decode)
- [ ] **9.9** iOS — confirm `NSFileProtectionComplete` is applied (the
      file becomes inaccessible while the device is locked)

**Known gap (revisited):** Phase 14 ships the writer + UI; the spec §16
strict signal-safety items (POSIX `open()` / `write()` / `close()` with
stack-allocated buffers on iOS, NDK `sigaction()` handler on Android)
are deferred to Phase 15. Test 9.2 today does not catch native crashes
on Android; this gap is documented in the README's *Known limitations*.

Acceptance: The capture pipeline works for app-language uncaught
exceptions; share sheet + audit chain are correct; 30-day sweep runs.

---

## 10. RTL verification

Required before v0.1 ships even though no RTL languages are active until
v1.1 (spec §11 RTL-readiness mandate).

- [ ] **10.1** iOS — Xcode scheme → Edit Scheme → Run → Options →
      Application Language → "Right-to-Left Pseudolanguage". Re-run the
      app. Every screen renders with the layout mirrored
- [ ] **10.2** Android — Developer options → Force RTL layout direction
      enabled. Relaunch the app. Every screen renders with the layout
      mirrored
- [ ] **10.3** Directional icons (back arrow, forward chevron) are
      auto-mirrored — back-chevrons in nav bars point right in RTL mode
- [ ] **10.4** `HStack` / `Row` orderings flip — multi-element rows
      appear right-to-left
- [ ] **10.5** No `.left` / `.right` modifiers — text and padding use
      `.leading` / `.trailing` semantics
- [ ] **10.6** No `padding(left = ...)` / `(right = ...)` — Compose uses
      `padding(start = ..., end = ...)`
- [ ] **10.7** Every screen sweep — Home, History (5 subsections, all 10
      detail and action screens), Settings (all editable rows), About,
      Legal, Crash logs list, Active screening (idle + active), Live
      override sheet, Review override form, Export confirmation, Reset
      device dialogs, every onboarding step
- [ ] **10.8** Text fields, pickers, and segmented controls flip
      correctly. The cursor lands on the trailing edge of a text field
      in RTL mode

Acceptance: Every screen renders without LTR-only assumptions.

---

## 11. Lock and auto-logout

Spec §9. Auto-logout is configurable 5 / 15 / 30 minutes.

**Known gap:** the shared `SessionTimer` is implemented in
`shared/src/commonMain/kotlin/com/malaria/SessionTimer.kt` (spec §6) but
is **not yet wired** into either platform's foreground services. As of
Phase 16, auto-logout is documented in the spec, configurable in
Settings, and audit-logged when changed (`auto_logout_changed`), but the
timer does not actually fire and re-lock the app. Phase 6 / 7 (identity)
landed without this wiring; Phase 8 iOS (camera) deferred it again.
Targeted for a Phase 15 follow-up.

The cases below should still be verified for the *manual lock* and
*scenePhase backgrounding* paths, which are wired today. The
auto-logout-fire-while-foreground case will fail until the timer is
wired — record it as "untested" rather than "failed" for v0.1.

- [ ] **11.1** Manual lock — Settings has a "Lock now" affordance (or
      equivalent) that immediately re-locks. Confirm the Home tab is
      hidden and the biometric prompt appears on next interaction.
      Acceptance: a `session_relocked_manual` audit entry is written
- [ ] **11.2** Background — send the app to background (Home button /
      gesture); on resume, the app returns to locked state if it was
      backgrounded long enough. Acceptance: a
      `session_relocked_background` audit entry is written if the
      threshold was met
- [ ] **11.3** Resume within window — backgrounding briefly and resuming
      should not re-lock; the previous session continues without
      re-auth
- [ ] **11.4** **Untested for v0.1 (gap):** auto-logout fires while app
      is foreground — once the `SessionTimer` is wired, the app should
      re-lock at the timeout boundary and write
      `session_relocked_timeout`. Today this case is not testable
      because the timer is not running
- [ ] **11.5** Biometric unlock after re-lock — biometric prompt appears,
      success writes `auth_success`, app returns to the same tab the
      user was on
- [ ] **11.6** Failed biometric — passcode / device credential fallback
      appears; failure writes `auth_failure`
- [ ] **11.7** No biometric available — device-credential prompt is the
      direct path; the app does not refuse to unlock if the device has
      *some* credential configured

Acceptance for v0.1: manual lock + scenePhase background paths work and
audit-log correctly. The auto-logout-fire case is recorded as a known
gap, not a regression.

---

## Known gaps versus spec §20

These items appear in the spec's manual test plan but are not yet
exercisable in the current build:

- **Auto-logout via `SessionTimer`** — placeholder until the timer is
  wired into the iOS scenePhase observer and Android `LifecycleObserver`.
  Phase 15 polish.
- **Hugging Face download with intermittent connectivity** — placeholder
  until the downloader is implemented. The 17 deferred remote models in
  `model_registry.json` show "Requires download (not in v1)" in
  Settings; no download code path exists.
- **In-app crash trigger** — no debug menu today; testers attach a
  debugger and throw manually. Adding a debug-only "Crash now" button is
  a small Phase 15 task.
- **Android NDK native-crash handler** — JVM exceptions are caught by
  `Thread.setDefaultUncaughtExceptionHandler`. Native crashes from JNI /
  TFLite-native bypass the writer. Phase 15.
- **iOS signal-safe crash writer** — current iOS writer uses Foundation
  (`FileManager`, `JSONEncoder`). Spec §16's strict POSIX path is Phase
  15.
- **Android live screening** — the entire flow 4 is blocked on Phase 3
  (TFLite export pipeline) for Android.
- **Localised UI** — onboarding chrome is English-only until Phase 12
  ships Swahili / French / Portuguese. Force-RTL verification (flow 10)
  still works with English text.
- **LICENSE body in onboarding** — current text is the placeholder
  pending the maintainer pasting the HL3-FULL text from
  [firstdonoharm.dev/build](https://firstdonoharm.dev/build). Test 2.3
  exercises the acknowledgement flow, not the text content.

---

## Reporting template

When running the plan against a build, copy this header and check off
each numbered row:

```
Build: <version> <sha>
Device: <iPhone/Android model + OS>
Tester: <initials>
Date: <YYYY-MM-DD>

Flow 1 (Bootstrap)
  [ ] 1.1  [ ] 1.2  [ ] 1.3  [ ] 1.4
Flow 2 (Onboarding Phase 1)
  [ ] 2.1  [ ] 2.2  ...
...
Notes / failures:
  <free text per failed item>
```

A passing v0.1 run does not require every box checked — items recorded as
"untested" against the documented gaps above are acceptable. Items
recorded as "failed" against a wired test path block the release.
