# Technical Glossary for Beginners

> **Status:** first-pass for v0.1. Audience: a clinical deployer (or new
> contributor) reading this codebase for the first time with medical or
> compliance background but limited mobile-dev or ML context. The entries
> are deliberately terse; each links out to the right deep-dive doc when
> one exists.

Cross-references throughout point to:
[`KMP_App_Specification.md`](../KMP_App_Specification.md) (§ numbers),
[`docs/SCHEMA.md`](./SCHEMA.md),
[`docs/COMPLIANCE.md`](./COMPLIANCE.md),
and [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md).

Terms are grouped by category, not alphabetised, so a reader can scan a
single section for what they need without scrolling through unrelated
vocabulary. Within a section, entries are ordered roughly from foundational
to project-specific.

## 1. Machine learning concepts

**CNN (Convolutional Neural Network).** The family of neural networks that
powers all the malaria models in this project. A CNN slides small learned
filters over an input image to extract features (edges, textures, shapes,
eventually parasite morphology) and stacks layers of those filters to
classify the input. The bundled `Malaria_BNLeaky_Keras` is a small custom
CNN; the 17 deferred remote models are transfer-learning variants of well-
known CNN backbones (MobileNet, EfficientNet, ResNet, etc.).

**Softmax.** The final activation that turns the CNN's raw logits into two
probabilities that sum to 1.0 — `parasitizedProb` and `uninfectedProb`. The
project stores both even though they sum to 1.0 because future multi-class
models (e.g., species-level classification) need the same field shape.

**Threshold.** The probability cutoff above which a prediction is labelled
`Parasitized` rather than `Uninfected`. Default 0.30 per spec §5; admin-
adjustable between 0.10 and 0.90 in Settings. Lowering the threshold
trades false positives for fewer false negatives — the right call for a
screening tool that hands the slide to a clinician for confirmation, not
for an autonomous diagnostic. See `Threshold.label()` in
`shared/src/commonMain/kotlin/com/malaria/Threshold.kt`.

**Gray zone.** A probability band (`0.30 ≤ parasitizedProb ≤ 0.70`) where
the model's confidence is treated as low and the prediction is flagged for
review. The bounds are inclusive on both sides per
`Threshold.shouldFlagForReview()` and mirror the iOS / Android
`RiskBandIndicator` UI. Predictions in the gray zone surface in the History
tab's **Flagged for review** subsection (spec §13).

**False negative.** The model predicts `Uninfected` on a sample that is in
fact parasitized. Clinically the **higher-cost** error class for malaria —
a missed parasitized slide can mean a delayed treatment for a treatable
infection. The default threshold of 0.30 (rather than the naive 0.50)
deliberately biases toward catching these at the cost of more false
positives.

**False positive.** The model predicts `Parasitized` on a sample that is in
fact uninfected. Lower-cost than a false negative in this screening
context: a flagged slide gets a closer look by the microscopist and is
either confirmed or overridden. This is precisely the override flow in
spec §12.

**Parasitized vs uninfected.** The two output classes of the v1 model.
"Parasitized" means the image contains a red blood cell hosting a *Plasmodium*
parasite at some stage of its lifecycle; "uninfected" means it does not.
The training dataset (NIH Malaria Cell Images) does not distinguish
species, so neither does v1. Species-level classification is a v2 question
(spec §24).

**Core ML.** Apple's on-device inference framework. Models are bundled as
`.mlpackage` directories (a bundle of model weights + metadata + compute
plan). The iOS `CoreMLClassifier` actual uses Vision (`VNCoreMLRequest`) to
run inference; Apple's runtime picks the optimal backend (CPU / GPU / ANE)
per device.

**TFLite / LiteRT.** Google's on-device inference runtime, now formally
renamed **LiteRT** in 2024 (the older `TFLite` name is still widely used
in code and is retained throughout this project). Models are bundled as
`.tflite` files. The Android `TFLiteClassifier` actual will use LiteRT
once Phase 3 (TFLite export pipeline) lands; Phase 3 is currently blocked
on the Part 7B notebook work in `Malaria_Detection_Detailed_Analysis.ipynb`.

**On-device inference.** The model runs entirely on the user's phone — no
network call to a cloud inference endpoint. This is a clinical-correctness
property (the app works offline in a rural clinic) and a privacy property
(no slide image ever leaves the device). It is also why model size and
quantization matter: a 200 MB model on a midrange phone is a non-starter.

**Model quantization.** Reducing the numeric precision of the model
weights (e.g., from `float32` to `int8`) to shrink the file and speed up
inference at a small accuracy cost. Several of the 17 deferred remote
models are quantized variants of an `fp32` baseline. The project tracks
per-model accuracy and size in `model_registry.json` so the admin can
trade off informedly.

**`.mlpackage`.** The Core ML model bundle format introduced in 2021,
superseding the older `.mlmodel`. A directory rather than a single file;
contains a manifest, the compiled model, and analytic metadata. Bundled
under `iosApp/Resources/Models/` and consumed by `CoreMLClassifier`.

**ANE (Apple Neural Engine).** A dedicated neural-network accelerator on
Apple silicon (iPhone, iPad, Mac). Core ML can dispatch ops to ANE when
the model and op set are compatible; the Phase 2 measurement of ~25 ms per
inference on iPhone simulator was without ANE. On a real iPhone with ANE
the latency is typically lower; the spec budget per spec §5 is < 200 ms,
so there's wide headroom.

**GPU/NPU delegate (Android).** The LiteRT equivalent of routing ops to a
non-CPU accelerator. Pixel devices route to a Tensor NPU (recent models)
or GPU (older models); other Android OEMs ship various NPU vendors.
LiteRT's delegate API handles the dispatch; the app does not pick a vendor.

**Inference latency (`inferenceMs`).** Wall-clock time from
`Classifier.classify(image)` entry to result return, measured in
milliseconds. Stored on every `Prediction` row for monitoring. See
`Prediction.inferenceMs` in `docs/SCHEMA.md`.

**Image hash (`imageHash`).** SHA-256 of the *preprocessed* input bytes
(128×128 RGB, normalised). Used for duplicate detection in the
"mark-as-duplicate" flow (spec §13). The original captured frame is
discarded; only the hash persists.

## 2. Mobile platform terms

**KMP (Kotlin Multiplatform).** JetBrains' Kotlin-language toolkit for
sharing pure-logic code across iOS, Android, JVM, and other targets. This
project uses KMP for the shared module (threshold logic, session grouping,
permissions, retention policy, model registry, image preprocessing,
classifier `expect` declaration). UI and persistence are **not** shared —
see `docs/ARCHITECTURE.md`.

**XCFramework.** Apple's binary framework format that ships multiple
platform slices (iOS device, iOS simulator, macOS) in one bundle. The
shared module's `assembleSharedReleaseXCFramework` Gradle task produces
the XCFramework consumed by `iosApp` via Swift Package Manager.

**AAR (Android Archive).** Android's binary library format (essentially a
JAR plus Android resources). The shared module's Android target is
consumed by `androidApp` via `implementation(project(":shared"))` and
materialises as an AAR at build time.

**expect/actual.** Kotlin's mechanism for declaring a type or function in
`commonMain` (an `expect`) and providing platform-specific implementations
(`actual`) in `iosMain` and `androidMain`. The canonical example in this
project is `expect class Classifier` in `shared/.../Classifier.kt` with
the iOS actual wrapping Core ML and the Android actual wrapping LiteRT.

**SwiftData.** Apple's iOS 17+ persistence framework, layered on Core Data
internals but with a Swift-native API. The four entities in `docs/SCHEMA.md`
are SwiftData `@Model` classes under `iosApp/Models/`.

**`@Model` (SwiftData).** A Swift macro that marks a class as a SwiftData
entity — the equivalent of Android Room's `@Entity`. The fields of the
class become persistent properties; SwiftData generates the storage schema
at runtime from the class definition.

**`@Observable` (Swift).** A Swift macro (iOS 17+) that makes a class
participate in SwiftUI's view-update graph. Used pervasively in this
project's iOS code (`AuthGate`, `CameraService`, `PredictionStore`, etc.)
in lieu of `ObservableObject` + `@Published`.

**Room.** Google's official Android persistence library, layered on
SQLite. The four entities in `docs/SCHEMA.md` are Room `@Entity` classes
under `androidApp/.../data/entities/`. Room generates DAO implementations
at compile time from the `@Dao`-annotated interfaces in
`androidApp/.../data/dao/`.

**SQLCipher.** A SQLite extension that encrypts the database file with
AES-256. This project uses it on Android for at-rest encryption of the
Room database, with the SQLCipher passphrase itself encrypted by an
Android Keystore key. See `docs/COMPLIANCE.md` § "Android encryption-at-
rest detail".

**Compose (Jetpack Compose).** Google's modern declarative UI toolkit for
Android, conceptually similar to SwiftUI. This project uses Compose with
Material 3 Expressive for all Android UI. See `docs/ARCHITECTURE.md` § 3
for why UI is not shared across platforms.

**SwiftUI.** Apple's declarative UI framework. This project uses SwiftUI
with iOS 26's Liquid Glass design language for all iOS UI.

**`NSFileProtectionComplete`.** The strongest iOS file-protection class.
Files marked with it are encrypted on disk and become inaccessible while
the device is locked (the encryption key is wiped from memory at lock).
Applied to the SwiftData store and the crash log directory. See
`docs/COMPLIANCE.md`.

**Android Keystore.** Google's hardware-backed key storage on Android.
Keys generated through the Keystore API are bound to either a Trusted
Execution Environment (TEE) or, where available, a dedicated security
chip (StrongBox). The app holds a *reference* to the key but never sees
the raw bytes.

**StrongBox.** A discrete hardware security chip available on recent
Pixel and select other Android devices. Where present, the Keystore is
asked to provision keys *in* StrongBox; falls back transparently to the
TEE on devices without it. Per `SecureKeyStore.kt` in
`androidApp/.../data/`, the project requests StrongBox first and falls
back to software-backed otherwise.

**Secure Enclave.** Apple silicon's hardware security coprocessor. Keys
in the Secure Enclave never leave it — operations like sign / decrypt
happen *inside* the enclave with only the result returned to the app.
The biometric (Face ID / Touch ID) match also runs in the Secure Enclave,
which is why Face ID unlocks the project's `LAContext` gate without ever
exposing facial geometry to the app.

**Biometric prompt.** The native OS dialog that presents the device's
biometric (Face ID, Touch ID, fingerprint, face unlock) for the app to
gate sensitive actions. The project uses it on app launch (the
`AuthGate`), before editing protected settings, before recording a review
override, and before the reset-device flow. iOS uses `LAContext`; Android
uses `androidx.biometric.BiometricPrompt` with
`BIOMETRIC_STRONG | DEVICE_CREDENTIAL`.

## 3. Compliance and governance terms

**HIPAA (Health Insurance Portability and Accountability Act).** US
federal law governing the handling of Protected Health Information (PHI).
A deployer using this software in a US clinical setting takes on HIPAA
conformance work — see `docs/COMPLIANCE.md`. The `US_HIPAA` value of the
`Jurisdiction` enum drives the displayed retention floor (6 years).

**GDPR (General Data Protection Regulation).** EU regulation on personal
data processing. The project supports `EU_GDPR_DE`, `EU_GDPR_FR`, and
`EU_GDPR_GENERIC` jurisdictions, each with its own retention floor (10 /
20 / 10 years).

**MDR (Medical Device Regulation, EU 2017/745).** The EU framework that
classifies software-as-medical-device. The maintainer of this project does
**not** pursue MDR certification; the `NOTICE` file and the in-app medical-
device disclaimer make this explicit. A deployer who wants regulatory
clearance does the conformity assessment work themselves.

**SaMD (Software as a Medical Device).** The regulatory category that
covers software whose function is medical. Whether *this* software is a
SaMD depends on the deployer's regulatory analysis and the framing they
give the tool ("decision support" vs "diagnostic"). The maintainer frames
it as decision support and ships it as a research prototype.

**Lawful basis (GDPR Art. 6).** The legal ground under which personal data
is processed. The project's onboarding captures one of three:
`EXPLICIT_CONSENT`, `VITAL_INTERESTS`, `HEALTH_PROVISION`. Captured at
admin provisioning, persisted to `ConsentRecord`, and shown read-only in
Settings.

**Jurisdiction.** The legal regime the deployer operates under. The
`Jurisdiction` enum has six canonical values: `US_HIPAA`, `EU_GDPR_DE`,
`EU_GDPR_FR`, `EU_GDPR_GENERIC`, `KE_DPA` (Kenya Data Protection Act),
`OTHER`. The selection drives the **displayed** retention floor; the
project does not auto-delete on the floor (advisory only per spec §17).

**Audit log.** The append-only `AuditEntry` table that records every
sensitive action the app takes — who, what, when, against what resource.
The canonical action vocabulary lives in `docs/SCHEMA.md` § "Canonical
audit action vocabulary"; the writer is `AuditLog.write(...)` on iOS and
Android.

**Append-only.** A database table that the application code never UPDATEs
or DELETEs against. The audit log is append-only by code convention: the
repositories expose only `write(...)`, not mutation or deletion methods.
A future deployer adding chain-hashing inherits the same convention.

**Chain-of-custody.** The traceable history of who handled a record
during its life. The project's audit log is the chain of custody for every
prediction, override, and configuration change. Reset device preserves
prior audit entries precisely to keep the chain unbroken (spec §10).

**DPIA (Data Protection Impact Assessment).** GDPR Art. 35's required
analysis for high-risk processing. The project does **not** ship a DPIA;
a deployer in a GDPR jurisdiction produces their own. See
`docs/COMPLIANCE.md` § "What v1 does not implement".

**BAA (Business Associate Agreement).** HIPAA's required contract with
any third party that handles PHI on your behalf. The project ships no
third-party services, so the BAA surface is small; a deployer who adds a
cloud tier picks up the BAA work.

**Retention.** How long records are kept before deletion. The
`RetentionPolicy` module returns the **displayed** minimum years per
jurisdiction (`US_HIPAA` 6, `EU_GDPR_DE` 10, `EU_GDPR_FR` 20,
`EU_GDPR_GENERIC` 10, `KE_DPA` 7, `OTHER` 6). The app surfaces these in
Settings but never auto-deletes — that is a deployer-policy decision.

**Hard delete vs soft delete.** A *hard* delete physically removes the
row from the database; a *soft* delete sets a `deletedAt` column and
hides the row from queries. The project implements neither in v1 (no
deletion path exists). A deployer with a strong right-to-erasure
obligation adds hard delete to their fork — see `docs/COMPLIANCE.md` §
"Hard-delete pattern".

**Pseudonymous identifier.** A stable identifier that is not directly
linked to a person's identity (no name, no PHI). The `actorId` UUID on
`ClinicianProfile` is the project's only clinician identifier; the app
never sees a clinician's name. The 2-character `initials` field is
optional and the deployer's call.

**Clinician override (live vs review).** The two override contexts on a
`Prediction`. **Live override** is the 2-tap in-screening flow (spec §12)
— verdict + reason, no biometric, no notes — designed to be friction-free
during active screening. **Review override** is the deliberate post-hoc
flow from History — verdict + reason + initials + checkbox + biometric
prompt + `contextReviewed = true`. Both write a single `override_recorded`
audit entry with `overrideContext` set to `"live"` or `"review"`.

**Session grouping.** The project's rule for grouping predictions into
sessions. A 30-minute implicit gap rule: if the next prediction arrives
within 30 minutes of the previous one in the same session, it joins the
same session; otherwise a new session starts. The grouping is computed,
not user-driven. See `SessionGrouping.kt` in `shared/`.

**Decision-support framing.** The discipline of presenting every model
output as advisory, never as a verdict. Concretely: every screen that
shows a prediction also shows an override affordance; the medical-device
disclaimer is acknowledged at onboarding and re-surfaced in Settings; the
copy throughout the UI avoids verbs like "diagnose". Spec §12 codifies the
framing requirement.

## 4. Software development terms

**Gradle.** The build tool driving both the shared module and the Android
app. Top-level `build.gradle.kts` + per-module `build.gradle.kts` (shared,
iosApp wrapper, androidApp). `./gradlew :shared:check` runs the shared
module tests across all targets.

**`xcodegen` (not currently used).** Some KMP repos generate
`*.xcodeproj` from YAML; this project keeps the Xcode project hand-
maintained in `iosApp/iosApp.xcodeproj/` because the surface area is small
and the project structure rarely shifts.

**FileProvider (Android).** The standard Android mechanism for exposing
internal-storage files (the SwiftUI equivalent is `UIActivityViewController`
with a `URL`). Used to share export bundles and crash logs via
`Intent.ACTION_SEND` without exposing raw paths. See `AndroidManifest.xml`
and `res/xml/file_paths.xml` for the file paths exposed.

**BiometricPrompt (Android).** `androidx.biometric.BiometricPrompt` is the
project's Android biometric API. The suspend wrapper `BiometricPrompter`
in `androidApp/.../auth/` makes it ergonomic from coroutines. Requires the
hosting Activity to be a `FragmentActivity` — `MainActivity` is upgraded
from `ComponentActivity` to `FragmentActivity` for this reason.

**ModelContainer / ModelContext (SwiftData).** The two top-level SwiftData
types. The `ModelContainer` owns the on-disk store; the `ModelContext` is
the read/write handle the app uses. `ModelContainerFactory` in
`iosApp/Models/` wires up the production container with
`NSFileProtectionComplete`.

**CompositionLocal (Compose).** Compose's dependency-injection mechanism.
A `CompositionLocal<T>` is provided at a high point in the composition
tree and read from descendants without prop-drilling. The project's
`AppLocals.kt` registers `LocalAuthGate`, `LocalAuditLog`,
`LocalPredictionStore`, `LocalSettingsStore`, etc.; views read what they
need via `LocalAuditLog.current`.

**`EnvironmentValues` / `@Environment` (SwiftUI).** SwiftUI's analogous
DI mechanism. The composition root constructs services and injects them
via `.environment(...)`; views read with `@Environment(AuthGate.self)
var authGate`. See `MalariaDetectorApp.swift`.

**cinterop.** Kotlin/Native's mechanism for calling C and Objective-C
APIs from Kotlin. Not currently used in this project (the iOS classifier
actual is in Swift, not Kotlin), but the mechanism exists in the shared
module's iOS target and is reserved for future work that needs to reach
into a C library from `commonMain`.

## 5. This project's vocabulary

**Audit action.** A canonical lowercase-snake string identifying what
happened: `prediction_created`, `auth_success`, `override_recorded`,
`device_reprovisioned`, etc. The full vocabulary is in `docs/SCHEMA.md` §
"Canonical audit action vocabulary". Strings are stored in canonical
English regardless of UI locale (spec §8). Adding a value is a versioning
event; removing one is breaking and not done.

**OverrideReason (the five canonical values).** The reason the clinician
gave for overriding a model prediction. The enum (spec §5) has exactly
five values; their canonical strings are stored regardless of UI locale.

- `image_quality` — slide image is too blurry, dim, or framed wrong to be
  the basis of a reliable prediction.
- `atypical_morphology` — the slide content is unusual (mixed infection,
  rare stage, artefact) in a way the training data did not represent.
- `model_false_positive` — model said `Parasitized`, clinician judges
  `Uninfected`.
- `model_false_negative` — model said `Uninfected`, clinician judges
  `Parasitized`.
- `other` — any reason not covered by the above; the optional `notes`
  field is the channel for explanation.

The five values are clinical concepts and are explicitly flagged for
clinical-advisor review before v1.0 ships (spec §11 footnote). Display
strings stay English-only on both platforms (spec §15) even when UI is
otherwise localised — reviewers and exporters must see the same
vocabulary.

**Jurisdiction (the six canonical values).** Stored as canonical
lowercase-snake strings: `us_hipaa`, `eu_gdpr_de`, `eu_gdpr_fr`,
`eu_gdpr_generic`, `ke_dpa`, `other`. Selected at admin onboarding;
persisted to `ConsentRecord`. Drives the displayed retention floor in
Settings (advisory only — the app does not auto-delete).

**LawfulBasis (the three canonical values).** Stored as canonical
lowercase-snake strings: `explicit_consent`, `vital_interests`,
`health_provision`. Selected at admin onboarding; persisted to
`ConsentRecord`. Shown read-only in Settings. The enum corresponds to
GDPR Art. 6 lawful bases the project considered most applicable to
clinical-screening contexts.

**Session (implicit).** A bag of predictions grouped by the 30-minute gap
rule (see *session grouping*). Not user-selected — the rule runs in code
and assigns a `sessionId` UUID on every new prediction. The user can
later relabel a session with an ASCII-only free-text label (spec §13).

**Active screening session (UI state).** A foreground UI state, **not**
the same as a session in the persistence sense. While the user is on the
Home tab with the camera running, the app is in "active screening" mode;
the bottom Capture button is enabled, the prediction overlay is
mounted, scenePhase backgrounding and `AuthGate` lock both call
`cameraService.stop()` per spec §11. The user explicitly leaves the
state by tapping "End session" on iOS (Android live screening is blocked
on Phase 3). Multiple "active screening session" UI states can roll up
into a single persisted session via the gap rule.

**Active screening UI state vs persisted session.** A common point of
confusion. Persisted `sessionId` is determined by the gap rule and lives
on every `Prediction` row. The UI's "active screening" toggle is just
which screen is foregrounded with the camera live. Two predictions
captured in two separate Capture-button taps within 30 minutes are in
the *same* persisted session even though the UI may have idled between
them.

**Provisioned-unclaimed.** The intermediate device state after the admin
finishes Phase 1 onboarding but before any microscopist claims the device
in Phase 2. The Home and History tabs are hidden; the device displays a
"Device provisioned for [Clinic] — awaiting microscopist claim" screen.
The reset-device flow returns the device to this state (spec §10).

**Phase 1 / Phase 2 onboarding.** The two-stage device-claim flow. Phase
1 (admin) sets up clinic, jurisdiction, lawful basis, inference policy,
admin biometric. Phase 2 (microscopist) is the per-microscopist claim:
initials, microscopist biometric, orientation walkthrough. A single
deployer doing both phases back-to-back is a supported path.

**Reset device.** The Settings → Data management flow that returns the
device to `provisioned-unclaimed`. Wipes the clinician row; preserves
predictions and audit history as chain-of-custody. Writes a
`device_reprovisioned` audit entry with the wiped `actorId` in metadata.
See spec §10 and `ResetDeviceCoordinator.performReset()`.

**Export bundle.** A signed ZIP containing `export.json` (all
predictions, audit entries, clinician profile, consent records as
canonical JSON) and `README.txt`. HMAC-SHA256 signature; byte-identical
between platforms for the same content. Spec §14; Phase 13 implementation
is end-to-end on both platforms.

**Crash log.** A structured `{incident-uuid}.json` written by the app on
an uncaught exception. Contains app/OS/device-class metadata, the stack
trace, the last 50 audit action canonical strings, a rough memory
readout, and an unlocked/locked flag — and *nothing else* (spec §16
forbids prediction data, override notes, initials, actor UUIDs, image
hashes, clinic config, consent records). Auto-expires after 30 days.

**Bundled model vs remote model.** The bundled model
(`Malaria_BNLeaky_Keras`) ships inside the app binary and is available
offline from the moment of install. The 17 remote models are listed in
`model_registry.json` for future download from Hugging Face — the
downloader is deferred (see `Known limitations` in the README). v1 ships
the bundled model only.

**Decision-support disclaimer.** The text shown at onboarding (medical-
device disclaimer step) and re-surfaced in Settings → Legal. Flagged for
clinical-advisor review before v1.0 alongside the `OverrideReason` enum
values.

**`provisioned-unclaimed` vs `complete`.** Two values of
`OnboardingState.phase`. `provisioned-unclaimed` is the in-between state
post-Phase-1 / pre-Phase-2; `complete` is the operational state where
Home and History tabs are visible. The composition root mounts
`OnboardingFlow` vs `RootView` / `RootScreen` based on this flag.

---

## See also

- [`KMP_App_Specification.md`](../KMP_App_Specification.md) — full spec
- [`docs/SCHEMA.md`](./SCHEMA.md) — canonical persistence schema and
  audit action vocabulary
- [`docs/COMPLIANCE.md`](./COMPLIANCE.md) — compliance posture, what v1
  implements, deployer responsibilities
- [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md) — three-layer architecture
  and shared-module contents
- [`docs/MANUAL_TEST_PLAN.md`](./MANUAL_TEST_PLAN.md) — what cannot be
  automated
