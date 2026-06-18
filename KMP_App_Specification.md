# Malaria Detector — Application Specification

**Project:** `malaria-detector`
**Platforms:** iPhone (iOS 26+) and Android phone (16+ / API 36+), feature-parity first-class targets. iPad is incidental — the iOS build runs on iPadOS but is not designed, tested, or claimed to support iPad form factors.
**Architecture:** Kotlin Multiplatform for business logic; native UI per platform; native persistence per platform
**iOS distribution:** Swift Package Manager (no CocoaPods)
**License:** Hippocratic License 3.0 with explicit medical-device disclaimer
**Distribution:** Open-source on GitHub, model artifacts on Hugging Face
**Status:** Specification — pre-implementation
**Version target:** v1.0 is reachable at maintainer discretion when the implementation is judged stable and the spec deliverables are complete. The medical-device disclaimer in `NOTICE` carries the project's safety framing in lieu of an external clinical-advisor sign-off gate. (Prior versions of this spec described a clinical-advisor review as the v0.x → v1.0 gate; that requirement has been removed from the project scope — see §24.)

This document is the single source of truth for what gets built. Both iOS and Android are first-class targets at feature parity. Every screen, every audit action, every persistence behavior is specified for both platforms. Where the platforms force structural differences (encryption mechanisms, biometric APIs, camera frameworks), each section gives the iOS approach and the Android approach side-by-side.

Where the spec says "deferred to v2," that means the choice was considered and explicitly excluded from v1, not "we forgot."

---

## Table of Contents

1. [Project framing](#1-project-framing)
2. [License and distribution](#2-license-and-distribution)
3. [Repository structure](#3-repository-structure)
4. [Architectural overview](#4-architectural-overview)
5. [Shared module: what's in it, what's not](#5-shared-module-whats-in-it-whats-not)
6. [Data flow: one screening, end to end](#6-data-flow-one-screening-end-to-end)
7. [Model assets and distribution](#7-model-assets-and-distribution)
8. [Persistence and audit](#8-persistence-and-audit)
9. [Identity, authentication, and authorization](#9-identity-authentication-and-authorization)
10. [Onboarding](#10-onboarding)
11. [Screen-by-screen behavior](#11-screen-by-screen-behavior)
12. [Override flow](#12-override-flow)
13. [Sessions and history](#13-sessions-and-history)
14. [Export](#14-export)
15. [Localization](#15-localization)
16. [Crash logs](#16-crash-logs)
17. [What is explicitly NOT in v1](#17-what-is-explicitly-not-in-v1)
18. [Compliance posture](#18-compliance-posture)
19. [Build configuration](#19-build-configuration)
20. [Testing strategy](#20-testing-strategy)
21. [CI/CD](#21-cicd)
22. [Phased build plan](#22-phased-build-plan)
23. [Public launch checklist](#23-public-launch-checklist)
24. [Open questions](#24-open-questions)
25. [References](#25-references)

---

## 1. Project framing

This is a research-prototype open-source application. It is not a medical device. It will not be submitted for FDA clearance, EU MDR conformity assessment, or any equivalent regulatory pathway under the maintainer's direct effort. The application's purpose is twofold:

- Demonstrate, on real iOS and Android hardware, the on-device inference pipeline produced by the malaria-detection ML notebook (Part 7).
- Provide a credible architectural blueprint that a downstream deployer (clinic, NGO, research institution) could adapt to a regulated deployment by adding the compliance work that v1 deliberately stops short of.

The maintainer's deliverable is two functioning apps — iOS and Android — built to feature parity from a shared Kotlin Multiplatform core. The principle throughout: implement what a clinical deployer cannot easily retrofit themselves; document what they can.

Encryption-at-rest ships on both platforms. The full chain-hashed cryptographic audit log does not. Single-clinician device with biometric gate ships on both. Multi-clinician role separation does not. Both platforms share the same shared Kotlin module, the same canonical schema (`SCHEMA.md`), the same audit-action vocabulary, the same export format. They differ only where the platforms make differing impossible.

---

## 2. License and distribution

### License: Hippocratic License 3.0

The repository ships under Hippocratic License 3.0 (the current published version as of 2023). This is a permissive license at its base, with additional clauses prohibiting use that:

- Violates the UN Universal Declaration of Human Rights
- Violates core ILO labor conventions
- Causes ecological harm above defined thresholds

The Hippocratic License is not OSI-approved. Some Linux distributions and some corporate legal teams will not accept it. For an open-source project aimed primarily at researchers, NGOs, and clinics in malaria-endemic regions, this trade-off is acceptable: easier downstream adoption by the target audience, occasional friction with package managers that the maintainer does not need to depend on.

### Medical-device disclaimer

Separately from the license, a `NOTICE` file at the repo root states:

> This software is provided for research and educational purposes only. It is NOT certified as a medical device under FDA SaMD, EU MDR 2017/745, the Kenya Health Act medical-devices regulations, or any other regulatory framework. It must NOT be used as the basis for clinical diagnostic decisions without conformance assessment by the deploying party under their local regulations. The authors and contributors disclaim all liability for clinical use. Deployers assume full responsibility for regulatory compliance, patient safety, and clinical validation in their jurisdiction.

This disclaimer is shown again during onboarding (admin phase) and the user must explicitly acknowledge it before the device can be provisioned.

### Distribution

- **Source code:** GitHub, public repository, semantic version tags
- **iOS binaries:** TestFlight for development builds; App Store distribution is not a maintainer commitment
- **Android binaries:** Google Play Internal Testing track for development builds; Google Play Store distribution is not a maintainer commitment
- **Model weights:** Hugging Face Hub at `huggingface.co/{maintainer}/malaria-detector-models`
- **Translations:** the project ships English-only; Crowdin scaffolding remains in the repo as a deployer-fork extension point (see §15)

---

## 3. Repository structure

A hybrid layout: monorepo for source code, separate hosting for binary model artifacts.

```
malaria-detector/                              ← main repo (public, code + docs only)
│
├── shared/                                    ← KMP shared module
│   ├── build.gradle.kts
│   └── src/
│       ├── commonMain/
│       │   ├── kotlin/com/malaria/
│       │   │   ├── domain/                    ← DTOs, enums, business rules
│       │   │   ├── ml/                        ← Classifier interface (expect)
│       │   │   ├── registry/                  ← ModelRegistry parsing
│       │   │   ├── preprocessing/             ← image preprocessing
│       │   │   ├── session/                   ← SessionGrouping
│       │   │   ├── permissions/               ← Permissions
│       │   │   ├── retention/                 ← RetentionPolicy
│       │   │   └── util/
│       │   └── resources/
│       │       └── model_registry.json
│       ├── iosMain/
│       │   └── kotlin/com/malaria/
│       │       ├── ml/                        ← CoreMLClassifier (actual)
│       │       └── platform/
│       └── androidMain/
│           └── kotlin/com/malaria/
│               ├── ml/                        ← TFLiteClassifier (actual)
│               └── platform/
│
├── iosApp/                                    ← Xcode project (SwiftUI + SwiftData)
│   ├── MalariaDetector.xcodeproj
│   ├── Package.swift                          ← SPM manifest referencing the shared XCFramework
│   ├── MalariaDetectorApp.swift               ← composition root: builds environment, wires services
│   ├── Models/                                ← @Model SwiftData classes
│   ├── Persistence/                           ← ModelContainerFactory, repositories
│   ├── Services/                              ← AuthGate, CameraService, Classifier wrapper (all @Observable)
│   ├── Environment/                           ← EnvironmentKey definitions + extension helpers
│   ├── State/                                 ← @Observable state holders for active-screening, onboarding
│   ├── Views/                                 ← SwiftUI screens (consume @Environment directly)
│   ├── Localization/                          ← Localizable.xcstrings (string catalogs)
│   └── Resources/
│       ├── Models/
│       │   └── Malaria_BNLeaky_Keras.mlpackage    ← only bundled model
│       ├── PrivacyInfo.xcprivacy
│       └── Info.plist
│
├── androidApp/                                ← Gradle module (Compose + Room)
│   ├── build.gradle.kts
│   └── src/main/
│       ├── kotlin/com/malaria/android/
│       │   ├── MalariaApplication.kt          ← composition root: builds CompositionLocal providers
│       │   ├── MainActivity.kt
│       │   ├── data/                          ← Room @Entity classes
│       │   │   ├── entities/
│       │   │   ├── dao/
│       │   │   └── MalariaDatabase.kt
│       │   ├── ui/                            ← Compose screens (consume CompositionLocal directly)
│       │   │   ├── screens/
│       │   │   ├── components/
│       │   │   ├── locals/                    ← CompositionLocal definitions + provider composables
│       │   │   ├── state/                     ← State holders for active-screening, onboarding
│       │   │   └── theme/
│       │   └── services/                      ← AuthGate, CameraService, Classifier wrapper
│       ├── assets/
│       │   └── models/
│       │       └── Malaria_BNLeaky_Keras.tflite   ← only bundled model
│       ├── res/
│       │   ├── values/                        ← strings.xml per locale
│       │   ├── values-sw/
│       │   ├── values-fr/
│       │   ├── values-pt/
│       │   └── xml/
│       │       ├── network_security_config.xml
│       │       └── data_extraction_rules.xml
│       └── AndroidManifest.xml
│
├── docs/
│   ├── KMP_App_Specification.md               ← this document
│   ├── SCHEMA.md                              ← canonical schema (SwiftData + Room)
│   ├── COMPLIANCE.md                          ← what's implemented vs deferred; hard-delete pattern
│   ├── Technical_Glossary_for_Beginners.md
│   ├── ARCHITECTURE.md
│   ├── MANUAL_TEST_PLAN.md
│   ├── CLOUD_TIER_REFERENCE.md                ← deferred-cloud-tier design for forkers
│   └── STORE_SUBMISSION.md                    ← release placeholder config; deployer responsibilities
│
├── .github/
│   ├── workflows/
│   │   ├── ci-shared.yml
│   │   ├── ci-ios.yml
│   │   ├── ci-android.yml
│   │   ├── release-ios.yml
│   │   └── release-android.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── SECURITY.md
│
├── LICENSE                                    ← Hippocratic 3.0
├── NOTICE                                     ← medical-device disclaimer
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── SECURITY.md
```

Model binaries live separately at `huggingface.co/{maintainer}/malaria-detector-models`. Only `Malaria_BNLeaky_Keras.mlpackage` (iOS) and `Malaria_BNLeaky_Keras.tflite` (Android) are bundled in the respective apps; the other 17 models per platform are downloaded on demand and cached indefinitely.

`model_registry.json` lives in the main repo (metadata, no binaries). Each entry's `huggingfaceRepo` and per-platform `*_path` fields point to the binary locations.

---

## 4. Architectural overview

The app has three architectural layers, applied identically across both platforms:

**Shared Kotlin (KMP module).** Business logic, domain types, ML inference interface, image preprocessing, session grouping, threshold logic, permissions, model registry. Compiles to:
- iOS: XCFramework consumed via Swift Package Manager
- Android: AAR consumed via Gradle module dependency

About 60% of the non-UI code lives here.

**Platform-native persistence.** SwiftData on iOS with `NSFileProtectionComplete`. Room on Android with SQLCipher AES-256, hardware-backed key in Android Keystore (StrongBox when available). The schemas are mirror-images of each other, with `docs/SCHEMA.md` as the canonical source. Schema drift is caught by a CI snapshot test on both platforms that verifies entity definitions match `SCHEMA.md`.

**Platform-native UI.** SwiftUI on iOS (Liquid Glass design language for iOS 26+), Jetpack Compose on Android (Material 3 Expressive for Android 16+). No shared UI code. No Compose Multiplatform. No ViewModels — see below.

The principle: anything that affects clinical correctness (threshold logic, session grouping, override permissions) lives in shared Kotlin and is written once. Anything that affects platform feel (UI, persistence ergonomics, biometric integration) is native. Both platforms ship at feature parity from v0.1 onward.

### No ViewModels — environment-injected services and observable state

The app uses each platform's native state-observation and dependency-injection primitives directly. There is no MVVM intermediate layer.

**iOS (SwiftUI 6 + Swift 6.1):**
- State holders are `@Observable` classes (the Observation framework, not legacy `ObservableObject`/`@Published`)
- Domain DTOs from the shared module are wrapped in thin `@Observable` state types only when reactivity requires it; otherwise consumed directly
- Services (`Classifier`, `PersistenceContext`, `AuthGate`, `CameraService`, `ModelRegistry`) are constructed once at the composition root in `MalariaDetectorApp` and injected via `@Environment`
- Custom `EnvironmentKey` types define each service's environment entry, with a `defaultValue` that crashes if accessed without proper setup (production code always sets it; tests and previews must set it explicitly)
- Views consume services via `@Environment(\.classifier)`, `@Environment(\.predictionStore)`, etc.
- No third-party DI framework (no Swinject, no Factory, no Resolver)

**Android (Compose 1.8+ + Kotlin 2.1):**
- State holders are plain Kotlin classes exposing `StateFlow` (or `State<T>` for simple values)
- Services (`Classifier`, `PersistenceContext`, `AuthGate`, `CameraService`, `ModelRegistry`) are constructed once at the composition root in `MalariaApplication` / `MainActivity` and injected via `CompositionLocal`
- Custom `CompositionLocal` keys defined per service, with `compositionLocalOf { error("...") }` as default (production sets via `CompositionLocalProvider`; tests provide test doubles)
- Composables consume services via `LocalClassifier.current`, `LocalPredictionStore.current`, etc.
- Coroutines scope to the screen lifecycle via `rememberCoroutineScope()` inside the composable, with explicit `Job` cancellation on `DisposableEffect` dispose
- Long-running operations (capture, classify, persist) launch into a service-owned `CoroutineScope` rather than the composable's scope, so they survive recomposition
- No third-party DI framework (no Hilt, no Koin, no Dagger)

**Why this works without ViewModels:**
- Configuration-change survival (Android rotation): state holders are constructed at the application or activity scope, not the composable; their lifetime exceeds any individual composition
- Process-death recovery (Android): `rememberSaveable` for transient UI state; durable state lives in Room and is reloaded on app launch
- Scoped coroutines: services own their scopes; composables observe results via `StateFlow.collectAsStateWithLifecycle()`

The trade-off: discipline about which scope launches each operation. A camera-capture coroutine launched from a composable would die if the composable left the composition mid-capture; instead, capture launches in the `CameraService`'s own scope, and the composable observes the result via `StateFlow`. The spec is explicit about this in Section 6 (data flow).

### Concurrency model

**iOS (Swift 6.1 strict concurrency):**
- All types in the shared module's iOS bindings are `Sendable` by default (Kotlin's generated Swift API is `Sendable`-friendly)
- All UI state access is `@MainActor`-isolated
- All side-effecting operations (camera, classify, persistence, network) are `async throws`
- No completion-handler-style APIs in app code; the camera and biometric framework completion handlers are wrapped in `withCheckedThrowingContinuation` at the service boundary
- Data-race-free by compile-time check; the spec does not need to repeat "thread-safe" qualifiers because the language enforces it

**Android (Kotlin 2.1 + coroutines 1.10):**
- All side-effecting operations are `suspend` functions
- UI state exposed as `StateFlow<T>`, observed via `collectAsStateWithLifecycle()` in composables
- Service-owned `CoroutineScope` instances are constructed with `SupervisorJob() + Dispatchers.Main.immediate` for UI-affecting services, `SupervisorJob() + Dispatchers.Default` for compute services (classification)
- Cancellation is structured: cancelling a parent job cancels all children deterministically
- Context parameters (Kotlin 2.1) used in shared module where they reduce boilerplate without obscuring dependencies — see Section 5

---

## 5. Shared module: what's in it, what's not

### In commonMain

**Domain DTOs.** Plain data classes carrying raw inference results and metadata. These are *not* persisted entities — each platform maps them to its own persistence types at the boundary.

```kotlin
// shared/src/commonMain/kotlin/com/malaria/domain/Prediction.kt
data class Prediction(
    val parasitizedProb: Float,
    val uninfectedProb: Float,
    val modelId: String,
    val timestamp: Instant,
    val inferenceMs: Long,
    val imageHash: String
)
```

The Prediction DTO has no `id`, no `sessionId`, no `flaggedForReview`. Those are computed or assigned by the platform's persistence layer.

**Enums and classifications.**

```kotlin
enum class ClinicianRole { ADMIN, MICROSCOPIST, OBSERVER }

enum class OverrideContext { LIVE, REVIEW }

enum class OverrideReason {
    IMAGE_QUALITY,
    ATYPICAL_MORPHOLOGY,
    MODEL_FALSE_POSITIVE,
    MODEL_FALSE_NEGATIVE,
    OTHER
}

enum class Jurisdiction { US_HIPAA, EU_GDPR_DE, EU_GDPR_FR, EU_GDPR_GENERIC, KE_DPA, OTHER }

enum class LawfulBasis { EXPLICIT_CONSENT, VITAL_INTERESTS, HEALTH_PROVISION }
```

The string serialization of these enums (`"image_quality"`, `"admin"`, etc.) is the canonical form stored in persistence on both platforms. UI translates to localized display strings at render time; the underlying data stays in canonical English.

**Threshold logic.** Identical between platforms — clinical-safety code, written once.

```kotlin
object Threshold {
    const val DEFAULT = 0.3f
    const val GRAY_ZONE_LOW = 0.3f
    const val GRAY_ZONE_HIGH = 0.7f

    fun label(parasitizedProb: Float, threshold: Float = DEFAULT): String =
        if (parasitizedProb >= threshold) "Parasitized" else "Uninfected"

    fun shouldFlagForReview(parasitizedProb: Float): Boolean =
        parasitizedProb in GRAY_ZONE_LOW..GRAY_ZONE_HIGH
}
```

**Session grouping.** Implicit 30-minute gap rule.

```kotlin
object SessionGrouping {
    const val SESSION_GAP_MINUTES = 30L

    fun assignSessionId(
        previousPredictionTimestamp: Instant?,
        previousSessionId: String?,
        now: Instant = Clock.System.now()
    ): String {
        if (previousPredictionTimestamp == null || previousSessionId == null) {
            return Uuid.random().toString()
        }
        val gap = now - previousPredictionTimestamp
        return if (gap.inWholeMinutes >= SESSION_GAP_MINUTES) {
            Uuid.random().toString()
        } else {
            previousSessionId
        }
    }
}
```

**Permissions.** Role-based action checks.

```kotlin
object Permissions {
    enum class Action {
        CHANGE_THRESHOLD,
        CHANGE_JURISDICTION,
        CHANGE_DEFAULT_MODEL,
        CHANGE_AUTO_LOGOUT,
        RESET_DEVICE,
        TRANSFER_ROLE,
        EXPORT_ALL_DATA,
        VIEW_AUDIT_LOG,
        CREATE_PREDICTION,
        OVERRIDE_PREDICTION,
        MARK_AS_DUPLICATE,
        RELABEL_SESSION
    }

    fun canPerform(role: ClinicianRole, action: Action): Boolean = when (action) {
        Action.CHANGE_THRESHOLD,
        Action.CHANGE_JURISDICTION,
        Action.CHANGE_DEFAULT_MODEL,
        Action.CHANGE_AUTO_LOGOUT,
        Action.RESET_DEVICE,
        Action.TRANSFER_ROLE -> role == ClinicianRole.ADMIN

        Action.EXPORT_ALL_DATA,
        Action.VIEW_AUDIT_LOG,
        Action.CREATE_PREDICTION,
        Action.OVERRIDE_PREDICTION,
        Action.MARK_AS_DUPLICATE,
        Action.RELABEL_SESSION -> role == ClinicianRole.ADMIN || role == ClinicianRole.MICROSCOPIST
    }
}
```

**Classifier interface.**

```kotlin
expect class Classifier(modelId: String) {
    suspend fun classify(image: ImageInput): Result<Prediction>
    fun close()
}

data class ImageInput(
    val rgbBytes: ByteArray,
    val width: Int,
    val height: Int
)
```

Each platform provides an `actual class Classifier`:
- iOS (`iosMain`): wraps Core ML and Vision framework
- Android (`androidMain`): wraps LiteRT (TensorFlow Lite) with GPU/NPU delegate

The `Classifier` interface is intentionally designed to support both edge (`CoreMLClassifier`, `TFLiteClassifier`) and cloud (`CloudClassifier`, not implemented in v1 or planned for v2) variants. A deployer adding cloud tier extends the architecture rather than modifying it — see `docs/CLOUD_TIER_REFERENCE.md` for the design sketch.

**Image preprocessing.** Resize-to-128, RGB-normalize, deterministic operations producing the byte array that gets hashed for the `imageHash` field.

**Retention policy.** Advisory only; not auto-enforced in v1.

```kotlin
object RetentionPolicy {
    fun minimumYears(j: Jurisdiction): Int = when (j) {
        Jurisdiction.US_HIPAA -> 6
        Jurisdiction.EU_GDPR_DE -> 10
        Jurisdiction.EU_GDPR_FR -> 20
        Jurisdiction.EU_GDPR_GENERIC -> 10
        Jurisdiction.KE_DPA -> 7
        Jurisdiction.OTHER -> 6
    }
}
```

The platform persistence layer reads this to inform the user ("Records on this device should be retained for at least 10 years"), but does not auto-delete records that exceed the retention period. Deletion is deliberate, deployer-driven, audited.

**Model registry.** Parses `model_registry.json` from bundled resources. Returns the list of available models with bundled/cached/remote status for the current platform.

### Not in commonMain (in platform-native code)

- Persistence entities (`@Model` on iOS, `@Entity` on Android)
- Database setup (`ModelContainer` on iOS, `RoomDatabase` on Android)
- Keystore / Keychain integration
- Biometric prompts
- Camera capture
- UI of any kind
- Crash log writing
- Hugging Face downloader (uses platform HTTP clients — `URLSession` on iOS, OkHttp on Android — for download progress and resumable transfer)
- Audit log writing

The audit log writing in particular is interesting: the audit log *schema* is defined in `SCHEMA.md` and consumed by both platforms identically, but the writer is platform-native because audit entries are persisted entities, and persistence is native.

---

## 6. Data flow: one screening, end to end

The flow is identical on both platforms; only the platform-specific bridging differs at the camera-capture and inference steps.

```
[User taps Capture in Home tab]
         │
         ▼
[Native CameraService captures frame]
   iOS:     AVCaptureSession → CMSampleBuffer → CVPixelBuffer
   Android: CameraX ImageAnalysis → ImageProxy → YUV → RGB
         │
         ▼
[Platform converts to ImageInput (shared DTO)]
   RGB bytes + width + height
         │
         ▼
[Shared: Preprocessor]
   Resize to 128×128, normalize 0-1, return preprocessed bytes
         │
         ▼
[Shared: Classifier.classify(image)]
   iOS:     CoreMLClassifier → VNCoreMLRequest → Apple Neural Engine
   Android: TFLiteClassifier → Interpreter.run() → GPU/NPU delegate
   Returns Prediction DTO (raw probabilities + metadata)
         │
         ▼
[Platform: map Prediction DTO to persistent entity]
   iOS:     SwiftData @Model
   Android: Room @Entity
   Apply Threshold.label() and Threshold.shouldFlagForReview()
   Apply SessionGrouping.assignSessionId() using last prediction
         │
         ▼
[Persist entity to encrypted store]
   iOS:     SwiftData → SQLite under NSFileProtectionComplete
   Android: Room → SQLCipher AES-256 with Keystore-managed key
         │
         ▼
[AuditEntry written: PREDICTION_CREATED]
   Same persistence boundary
         │
         ▼
[View observes via reactive query, re-renders inline]
   iOS:     SwiftUI View with @Query (SwiftData) — direct observation, no ViewModel
   Android: Composable collects StateFlow from PredictionStore service via
            collectAsStateWithLifecycle(); state holder owned by service, not composable
   Verdict + confidence shown to user
         │
         │  (in-memory image released)
         ▼
[Ready for next capture]
```

The clinical-safety steps (label assignment, flagging logic, session grouping) all happen in shared Kotlin. The persistence, UI, and platform integration happen natively. Identical decisions, identical audit trails, identical schema — different platform mechanisms underneath.

**Scope ownership in this flow:** the capture-classify-persist sequence is launched into the `CameraService`'s long-lived `CoroutineScope` (Android) or runs as an `async throws` task awaited from a `@MainActor` view event handler (iOS, where SwiftUI's structured-concurrency `.task` modifier handles cancellation on view disappearance). Neither platform ties the operation to the lifetime of an individual composable/view re-render — a recomposition mid-classify must not lose the prediction.

Every step produces an audit entry. Capture failure, classification failure, hash collision (unlikely but theoretically possible), session ID generation — each is logged on both platforms.

The image bytes exist only in memory between camera capture and `imageHash` computation. After the prediction is persisted, the image bytes are explicitly released. Nothing about the image survives beyond the hash unless the user (in some future v1.1) opts in to research contribution during the same session.

---

## 7. Model assets and distribution

### Bundled model

The Model 2 winner from the notebook's Part 2 (BatchNorm + LeakyReLU, 97.73% test accuracy) ships inside each app bundle in the appropriate native format:

- **iOS:** `Malaria_BNLeaky_Keras.mlpackage` (~16 MB)
- **Android:** `Malaria_BNLeaky_Keras.tflite` (~4–8 MB after TFLite conversion and int8 quantization)

The bundled model is the deployment-recommended model per the notebook's executive summary. It is always available offline. It cannot be deleted from the app.

### Remote models

The other 17 models from the notebook are hosted at `huggingface.co/{maintainer}/malaria-detector-models`. The repository contains both `.mlpackage` (iOS) and `.tflite` (Android) files for each architecture. Both formats are generated in Phase -1 of the build plan (Section 22) before code-writing begins, so v0.1 ships with full feature parity — every model available on every platform.

Models are downloaded on demand when a user selects one in the Model Picker. Downloaded models are cached:
- **iOS:** `Application Support/Models/` under `NSFileProtectionComplete`
- **Android:** `context.filesDir/models/` with file-level encryption via `EncryptedFile` from Jetpack Security

Cache persists indefinitely until the user explicitly clears it from Settings → Models → "Clear cache."

### `model_registry.json`

Bundled with the shared module at `shared/src/commonMain/resources/model_registry.json`. Schema:

```json
{
  "schemaVersion": "1.0",
  "models": [
    {
      "id": "BNLeaky_Keras",
      "displayName": "BN + LeakyReLU (Keras)",
      "framework": "Keras",
      "architecture": "BN+LeakyReLU",
      "filenameStem": "Malaria_BNLeaky_Keras",
      "inputSize": 128,
      "paramCount": 8484546,
      "testAccuracy": 0.9773,
      "bundled": true,
      "huggingfaceRepo": null,
      "iosPath": null,
      "androidPath": null,
      "iosExpectedSha256": null,
      "androidExpectedSha256": null,
      "iosFileSizeMb": 16.2,
      "androidFileSizeMb": 5.4,
      "description": "BatchNorm after every Conv layer + LeakyReLU activations. Stable training. The deployment-recommended model."
    },
    {
      "id": "EfficientNetB3_Keras",
      "displayName": "EfficientNetB3 (Keras) ★",
      "framework": "Keras",
      "architecture": "EfficientNetB3",
      "filenameStem": "Malaria_EfficientNetB3_Keras",
      "inputSize": 300,
      "paramCount": 12320528,
      "testAccuracy": 0.9823,
      "bundled": false,
      "huggingfaceRepo": "{maintainer}/malaria-detector-models",
      "iosPath": "Malaria_EfficientNetB3_Keras.mlpackage",
      "androidPath": "Malaria_EfficientNetB3_Keras.tflite",
      "iosExpectedSha256": "<hash>",
      "androidExpectedSha256": "<hash>",
      "iosFileSizeMb": 50.4,
      "androidFileSizeMb": 14.8,
      "description": "Two-phase fine-tuned EfficientNetB3. Best accuracy. Squeeze-excitation blocks focus on parasite stain."
    }
  ]
}
```

The platform reads the registry, filters by what it can use (iOS reads `iosPath` / `iosExpectedSha256`; Android reads the Android equivalents), and presents the unified list to the user.

### Model Picker UX

Identical on both platforms. The picker shows all 18 models grouped by framework (Keras / PyTorch). Each row displays:

- Model display name
- Test accuracy and parameter count
- Status badge:
  - "✓ Available offline" — bundled model
  - "✓ Available offline (cached)" — previously downloaded
  - "↓ Download (47 MB)" — not yet downloaded, online
  - "Requires internet" — not yet downloaded, offline

Tapping a "Download" model with connectivity triggers the download with progress. The download is resumable across app launches. A "Cancel" affordance lets the user abort.

Tapping a "Requires internet" model when offline shows a dialog explaining the situation and offering to retry when online.

Tapping any model that's available (bundled or cached) makes it the active model immediately. The choice is audited as `ACTIVE_MODEL_CHANGED`.

Settings → Models also shows total cache size and a "Clear cache" button (per-model and all-cached options). Clearing is audited.

### Downloader implementation

- **iOS:** `URLSession` with background download tasks. Resumable via `URLSessionDownloadTask.cancel(byProducingResumeData:)`. Downloads land in a temp directory, then move into the cache directory atomically.
- **Android:** OkHttp with `Range` headers for resumable transfer. WorkManager-orchestrated for resumability across process death.

SHA-256 verification against the registry's expected hash happens before the file is moved into the cache. A hash mismatch aborts the download and audits `MODEL_DOWNLOAD_HASH_MISMATCH`. The user sees "Download failed — file integrity check failed. Please try again."

---

## 8. Persistence and audit

### Approach

- **iOS:** SwiftData with `NSFileProtectionComplete` on the store file. The store is cryptographically unreadable when the device is locked. Apple manages the encryption key via the Secure Enclave.
- **Android:** Room with SQLCipher AES-256. Database key in Android Keystore, hardware-backed via StrongBox when available. SQLCipher's `SupportFactory` integrates with Room transparently.

Both approaches satisfy HIPAA §164.312(a)(2)(iv) encryption-at-rest. The iOS approach inherits OS-level encryption that's accessible whenever the device is unlocked; the Android approach maintains an app-specific encryption boundary even when the device is unlocked. This asymmetry is documented in `COMPLIANCE.md` — both meet the bar, but Android is slightly stronger in the device-unlocked threat model.

### Canonical schema (`SCHEMA.md`)

The schema is defined once in `docs/SCHEMA.md` and implemented twice: as SwiftData `@Model` classes on iOS and as Room `@Entity` classes on Android. Schema drift is caught by CI snapshot tests on both platforms — each platform's entity definitions are dumped to a normalized JSON representation and compared against `SCHEMA.md`. Any discrepancy fails the build.

#### Entity: `Prediction`

| Field | Type | Notes |
|-------|------|-------|
| id | String (UUID) | Primary key, generated client-side |
| sessionId | String (UUID) | Assigned by `SessionGrouping` |
| timestamp | Date / Instant | Capture time, UTC |
| modelId | String | Matches `model_registry.json` id |
| modelVersion | String | Hash of the model file at inference time — locks audit to exact weights |
| parasitizedProb | Double | Raw model output |
| uninfectedProb | Double | Raw model output |
| label | String | Computed via `Threshold.label()` |
| threshold | Double | Threshold at time of capture, from the device's locked policy |
| flaggedForReview | Bool | Computed via `Threshold.shouldFlagForReview()` |
| inferenceMs | Int | Latency, for monitoring |
| imageHash | String | SHA-256 of preprocessed input bytes |
| clinicianOverride | String? | Nullable; populated by override flow |
| overrideContext | String? | "live" or "review" |
| duplicateOfId | String? | Nullable; if set, this prediction is a duplicate of another |
| sessionLabel | String? | Free-text label applied to the session; mirrored on every prediction in the session for query simplicity |
| appVersion | String | At time of capture |
| osVersion | String | At time of capture |

Note: there is no `image` field, no `imageBlob`, no path to a stored image. Images do not persist on either platform.

#### Entity: `AuditEntry`

| Field | Type | Notes |
|-------|------|-------|
| id | String (UUID) | Primary key |
| seq | Long (auto) | Monotonic per-device sequence |
| timestamp | Date / Instant | When the action occurred |
| actorId | String | UUID of the device's clinician profile |
| actorRoleAtTime | String | Role snapshot at the time of action |
| action | String | Enum value, see action list below |
| resourceType | String? | "prediction", "model", "setting", etc. |
| resourceId | String? | UUID of the resource if applicable |
| metadataJson | String | Structured details, JSON-encoded |
| overrideContext | String? | "live" or "review", for OVERRIDE_RECORDED only |
| overrideReason | String? | Enum value, for OVERRIDE_RECORDED only |
| overrideNotes | String? | Free text, for OVERRIDE_RECORDED only |
| contextReviewed | Bool? | True only for review overrides |
| overrideActorInitials | String? | Free-text initials, for OVERRIDE_RECORDED only |
| appVersion | String | |
| osVersion | String | |

The audit log is **append-only** by application convention (no UPDATE or DELETE SQL ever issued against it by app code) on both platforms. Without cryptographic chain-hashing, this is a discipline rather than a technical guarantee. A deployer wanting tamper-evident audit logs adds chain hashing themselves; the architecture supports it cleanly (Section 18).

#### Audit action vocabulary

Identical on both platforms. Stored as canonical English strings regardless of UI locale.

Onboarding and lifecycle:
- `ADMIN_PROVISIONING_STARTED`
- `CLINIC_CONFIGURED`
- `INFERENCE_POLICY_SET`
- `ADMIN_BIOMETRIC_ENROLLED`
- `ADMIN_PROVISIONING_COMPLETED`
- `MICROSCOPIST_CLAIM_STARTED`
- `MICROSCOPIST_BIOMETRIC_ENROLLED`
- `MICROSCOPIST_CLAIM_COMPLETED`
- `DEVICE_REPROVISIONED`
- `ROLE_TRANSFERRED`
- `PROFILE_UPDATED`

Authentication:
- `AUTH_SUCCESS`
- `AUTH_FAILURE`
- `SESSION_UNLOCKED`
- `SESSION_RELOCKED_BACKGROUND`
- `SESSION_RELOCKED_MANUAL`
- `SESSION_RELOCKED_TIMEOUT`

Models:
- `MODEL_DOWNLOAD_INITIATED`
- `MODEL_DOWNLOAD_COMPLETED`
- `MODEL_DOWNLOAD_FAILED`
- `MODEL_DOWNLOAD_HASH_MISMATCH`
- `MODEL_CACHE_CLEARED`
- `ACTIVE_MODEL_CHANGED`

Inference:
- `PREDICTION_CREATED`
- `PREDICTION_VIEWED`
- `OVERRIDE_RECORDED`
- `PREDICTION_LINKED_AS_DUPLICATE`
- `SESSION_RELABELED`

Data management:
- `EXPORT_INITIATED`
- `EXPORT_COMPLETED`
- `EXPORT_FAILED`
- `CRASH_LOG_SHARED`

Configuration:
- `THRESHOLD_CHANGED`
- `DEFAULT_MODEL_CHANGED`
- `AUTO_LOGOUT_CHANGED`
- `LANGUAGE_CHANGED`

Adding a new action value is a versioning event documented in `SCHEMA.md`. Removing one would be a breaking change and is not done — old logs continue to contain the old values.

### What v1 implements (on both platforms)

- Full persistence schema (Prediction, AuditEntry, ClinicianProfile, ConsentRecord)
- At-rest encryption — `NSFileProtectionComplete` on iOS, SQLCipher AES-256 on Android
- Hardware-backed encryption keys — Secure Enclave on iOS, Android Keystore with StrongBox on Android
- Audit entry written for every action in the vocabulary above
- Append-only by convention
- Biometric gate on app launch (Section 9)
- Auto-logout after configurable inactivity (default 30 min, set during admin onboarding)
- Soft delete: predictions can be marked-as-duplicate; no hard delete from UI
- Retention policy displayed in Settings (per jurisdiction), not auto-enforced

### What v1 explicitly does not implement

These are documented in `COMPLIANCE.md` as deployer responsibilities:

- Cryptographic chain hashing of audit log
- Daily integrity verification of audit chain
- Per-jurisdiction auto-retention (auto-delete after N years)
- Hard-delete flow (even admin-authenticated)
- Tamper-detection / anomalous-access flagging
- Multi-clinician audit attribution (v1 is single-clinician with override-time initials only)

---

## 9. Identity, authentication, and authorization

### Identity model

A single-clinician device. One clinician profile per app installation. Identical model on both platforms:

```
ClinicianProfile {
    actorId: String (UUID, generated at onboarding, never PII)
    role: ClinicianRole (.admin | .microscopist | .observer)
    initials: String? (2 chars, optional, free text)
    enrolledAt: Date / Instant
    biometricEnrolled: Bool
}
```

The UUID is never linked to a real name inside the app. A clinic admin keeps a separate paper or system mapping (UUID → person) outside the app's scope. This keeps the app PII-free regardless of how the clinic chooses to track it externally.

Multi-clinician device support is a v2 commitment, not deferred indefinitely. The v1 single-clinician choice is documented as v1-specific architecture, with the v2 multi-clinician design sketched in Section 24. v1's schema does not pre-declare multi-clinician fields; v2 work is a proper schema migration.

The override flow captures multi-actor information at the only point where it matters clinically (the override event) via free-text initials. This is the v1 workaround for the single-clinician assumption.

**Imported actor IDs (v1.1 import flow).** When the v1.1 import feature lands, imported predictions preserve their original `actorId` rather than re-attributing to the importing clinician. The receiving device becomes effectively multi-actor for those imported rows, while remaining single-clinician for new predictions. Imported actor IDs are read-only — the `Permissions` module treats them as historical references; no actions can be performed *under* an imported actor identity. Audit entries for newly-imported predictions log both the importing clinician and the original device UUID.

### Authentication

Biometric gate on app launch. Both platforms use device biometric with passcode fallback:

- **iOS:** `LAContext.evaluatePolicy(.deviceOwnerAuthentication, ...)` — Face ID or Touch ID, with device passcode fallback
- **Android:** `BiometricPrompt` with `BIOMETRIC_STRONG | DEVICE_CREDENTIAL` authenticators — fingerprint or face (only Class 3 / `BIOMETRIC_STRONG`, no Class 2 / weak face unlock), with device credential fallback

If the device has no passcode/PIN/biometric configured at all, the app refuses to unlock and prompts the user to set one via system settings.

Session model (identical on both platforms):
- Authentication unlocks the app
- Session persists until: app backgrounded > 5 minutes, user manually locks via Settings, or auto-logout timer fires
- Configurable auto-logout (set by admin during onboarding): 5, 15, or 30 minutes of inactivity

The session timer runs in the shared Kotlin module:

```kotlin
class SessionTimer(
    private val timeoutMinutes: Int = 30,
    private val onTimeout: suspend () -> Unit
) {
    private var lastActivity: Instant = Clock.System.now()

    fun touch() { lastActivity = Clock.System.now() }

    suspend fun checkTimeout() {
        val elapsed = Clock.System.now() - lastActivity
        if (elapsed.inWholeMinutes >= timeoutMinutes) {
            onTimeout()
        }
    }
}
```

Each platform calls `touch()` on every relevant user interaction and `checkTimeout()` periodically (every 30 seconds via a timer, plus on every app foreground):
- **iOS:** scenePhase observer for foreground, Combine timer for periodic check
- **Android:** Lifecycle observer for foreground, WorkManager periodic check or coroutine-based timer

### Partial-lock model

Two zones in the app:

- **Always accessible** (no auth required): Settings tab, About tab. Configuration is visible; edits trigger biometric prompt.
- **Auth required**: Home tab (camera + active screening), History tab (predictions, audit log, exports).

This is enforced at the tab level on both platforms. The locked tabs render a "Tap to unlock" placeholder until authenticated. Once unlocked, both tabs become functional for the duration of the session.

### Authorization

Two-dimensional: role × action.

For a single-clinician device, the role determines what's in the UI:
- Admin: sees full inference policy editable in Settings; can re-onboard; can transfer role
- Microscopist: sees inference policy read-only; can capture and override; can export
- Observer: read-only access to History; cannot capture

The `Permissions` shared module is the single source of truth (Section 5). The UI on both platforms consults it before rendering editable controls.

### Fresh auth prompts

These actions trigger a fresh biometric prompt regardless of current session state, on both platforms:

- Export all data
- Reset device
- Transfer role
- Change threshold, default model, or auto-logout (admin only)
- Override a prediction during review (not during live screening — live override is too high-frequency to gate)

---

## 10. Onboarding

Two-phase. The phases can happen on different days, in different places, with different people present. Flow is identical on both platforms; only the OS-level biometric registration mechanism differs.

### Phase 1 — Admin provisioning

Performed by the clinic admin (typically IT lead or clinic manager). The device is fresh from the box or freshly reset.

Screens, in order:

1. **Language picker.** English, Swahili, French, Portuguese. Stored in `UserDefaults` (iOS) / `DataStore` (Android) — unencrypted; not sensitive. Persists across resets.
2. **Welcome.** What the app is, who it's for. One paragraph + "Continue."
3. **Hippocratic License acknowledgement.** Full license text shown. "I have read and accept" checkbox + Continue.
4. **Medical-device disclaimer acknowledgement.** Full `NOTICE` text shown. Checkbox + Continue.
5. **Clinic details.**
   - Clinic name (free text, required)
   - Jurisdiction picker (the `Jurisdiction` enum)
   - Lawful basis picker (the `LawfulBasis` enum) with one-line explanations
6. **Inference policy.**
   - Decision threshold (slider 0.0–1.0, default 0.3, with the "trade-off" explanatory copy)
   - Default model selection (picker from `model_registry.json`)
   - Auto-logout timeout (5 / 15 / 30 min picker)
7. **Admin biometric registration.** Native biometric prompt. This biometric authorizes future admin-only actions, never routine operation.
8. **Provisioning complete.** "Device provisioned for [Clinic name]. Hand to microscopist to complete setup."

After step 8, the device is in `provisioned-unclaimed` state. The Home and History tabs do not render — only Settings (read-only) and a banner "Setup incomplete. Tap to begin Phase 2."

Audit entries written during Phase 1: `ADMIN_PROVISIONING_STARTED`, `CLINIC_CONFIGURED`, `INFERENCE_POLICY_SET`, `ADMIN_BIOMETRIC_ENROLLED`, `ADMIN_PROVISIONING_COMPLETED`.

### Phase 2 — Microscopist claim

Performed by the field microscopist when they receive the device.

Screens, in order:

1. **Welcome.** "This device is provisioned for [Clinic name]. Continue?"
2. **Initials (optional).** 2-character free text or skip. Stored as the microscopist's display initials in audit reports.
3. **Microscopist biometric registration.** Native biometric prompt. This biometric becomes the routine unlock biometric.
4. **Quick orientation.** Three screens explaining: how to capture, how to override, how to lock. Skippable.
5. **Begin screening.** App transitions to operational state. Home tab becomes accessible.

Audit entries: `MICROSCOPIST_CLAIM_STARTED`, `MICROSCOPIST_BIOMETRIC_ENROLLED`, `MICROSCOPIST_CLAIM_COMPLETED`.

### Single-person deployment

When admin and microscopist are the same person, after Phase 1 step 8 the app offers: "You're the microscopist too? Continue Phase 2 now." Both biometric enrollments happen back-to-back; both register the same finger; the role check differentiates the two contexts going forward.

### Re-onboarding

Settings → Data management → "Reset device" (admin-authenticated, double-confirmed). Audit: `DEVICE_REPROVISIONED`. Database wiped (predictions, audit log entries for non-re-provisioning events, clinician profiles). Device returns to `provisioned-unclaimed` state — clinic-level config preserved (jurisdiction, threshold, default model), clinician-level wiped. The provisioning audit entries from the original Phase 1 are preserved as chain-of-custody for the device's history.

### Edge cases

- Phase 1 completed but device lost before Phase 2: replacement device starts fresh. No data was created.
- Microscopist transfers device to another microscopist mid-deployment: Settings → "Transfer to new microscopist" → fresh biometric (current microscopist) → device returns to `provisioned-unclaimed` → new microscopist completes Phase 2.
- Admin loses access (biometric no longer works): no recovery path in v1. The device must be reset via OS-level erase and re-provisioned from scratch. Documented in `COMPLIANCE.md` as a known limitation.

---

## 11. Screen-by-screen behavior

Identical functionality across both platforms. UI is platform-native: SwiftUI on iOS uses Liquid Glass (iOS 26 design language); Compose on Android uses Material 3 Expressive (Android 16 design language).

### RTL-readiness requirement

All UI is written RTL-ready from v1, even though v1 does not ship any RTL languages and the maintainer does not commit to shipping one. The discipline costs nothing during v1 development and means a deployer-fork that translates to an RTL language gets a layout that already flips correctly (see Section 24 — deployer-fork territory).

Concretely:

- **SwiftUI:** use `.leading` and `.trailing` everywhere, never `.left` or `.right`. Use `HStack` (which respects layout direction) rather than absolute positioning. Text alignment uses `.leading` (which flips correctly in RTL).
- **Compose:** use `Modifier.padding(start = ..., end = ...)` rather than `(left = ..., right = ...)`. Use `Arrangement.Start` not `Arrangement.Absolute.Left`. The framework's `LayoutDirection` does the flipping work.
- **Icons with directional meaning** (back arrows, forward chevrons): use SwiftUI's automatic mirroring on iOS, `AutoMirrored` icon variants on Android.
- **Verification:** the manual test plan (Section 20) requires running the app with "Force RTL" developer option enabled on both platforms before v0.1 ships, even with English text, to catch LTR-only assumptions early.

### Tab bar / bottom navigation

Four tabs, always visible. iOS uses `TabView` with a bottom tab bar; Android uses Material 3 `NavigationBar`.

- **Home** (locked when not authenticated): active screening
- **History** (locked when not authenticated): past predictions, audit log, data management
- **Settings** (always accessible): configuration; edits trigger biometric
- **About** (always accessible): version, license, credits

When not authenticated, Home and History show a "Tap to unlock" placeholder.

### Home tab

Layout depends on state.

**Idle state** (no active capture):
- Top: model picker affordance ("BN + LeakyReLU ★ Edit")
- Center: large camera preview area, currently dark
- Bottom: primary "Capture" button

Tapping Capture transitions to **active screening** state:
- Camera preview live (AVCaptureSession on iOS, CameraX `PreviewView` on Android)
- An overlay shows last prediction inline ("Parasitized 87%") if a prediction was just made
- "Override" affordance next to the prediction
- "End session" affordance (returns to idle)

Active screening continues until: user explicitly ends session, app backgrounded, auto-logout fires, or device is rotated to a state the app doesn't support (camera is portrait-only on both platforms).

Per-prediction flow during active screening:
1. User frames a cell in the preview
2. User taps Capture
3. Camera captures frame
4. Frame converted to ImageInput, passed to Classifier
5. Result rendered inline (~50-300ms depending on model)
6. User reads result, either taps Capture again for next cell or taps Override to disagree

### History tab

Five subsections, accessible via a list at the top:

- **Recent predictions** — newest first, paginated
- **Flagged for review** — `flaggedForReview = true` and `clinicianOverride IS NULL`
- **Sessions** — predictions grouped by session, with summary stats per session
- **Audit log** — chronological, filterable by action / actor / date range
- **Data management** — Export, Lock device, Reset device

Tapping a prediction opens its detail view (AI Analysis):
- Verdict + confidence
- Threshold at time of capture
- Model used
- Session it belongs to (link)
- Override status (link to override flow if not yet overridden)
- "Mark as duplicate of..." affordance

Session detail view:
- Session date and duration
- Total cells, parasitized count, mean confidence, gray-zone count
- Free-text label (editable, with PII warning)
- List of predictions in this session (link to each)

Audit log detail view:
- Action, timestamp, actor
- Full metadata JSON, formatted as readable key-value pairs
- For OVERRIDE_RECORDED entries: reason, notes, override actor initials

### Settings tab

Always accessible. Tapping an editable row triggers biometric prompt.

Sections:

- **Clinic**
  - Clinic name (read-only)
  - Jurisdiction (read-only)
  - Lawful basis (read-only)
- **Clinician profile**
  - UUID (read-only, copyable)
  - Role (read-only)
  - Initials (editable → biometric)
- **Inference**
  - Decision threshold (read-only for microscopist; editable → biometric for admin)
  - Default model (read-only for microscopist; editable → biometric for admin)
  - Auto-logout timeout (read-only for microscopist; editable → biometric for admin)
- **Models**
  - Bundled: Malaria_BNLeaky_Keras ✓
  - Downloaded: (cached models with sizes)
  - Available: (downloadable models)
  - Total cache size + "Clear all caches" → biometric
- **Language**
  - Picker (editable → biometric to prevent stranger-flips)
- **Legal**
  - Privacy policy (link)
  - Terms of service (link)
  - Decision-support disclaimer (link)
  - Open-source acknowledgements (link)
- **Crash logs** (Section 16)
  - Count
  - "Review and share" → biometric

### About tab

Always accessible. Static content:
- App name and version
- Build number
- Hippocratic License link
- Source code link
- Maintainer credits
- Contributor credits (linked to repo's CONTRIBUTORS file)

---

## 12. Override flow

Two contexts: live (during active screening) and review (from History tab, after the session ended). Different UX, different fields captured, different audit semantics. Identical on both platforms.

### Live override (2-tap during active screening)

Triggered: user taps "Override" next to the just-rendered prediction.

Screen 1 (immediate, full-screen modal):

```
The model said: PARASITIZED (87%)

Override to:
   [ Parasitized ]
   [ Uninfected  ]

(Tapping one transitions to screen 2)
```

Screen 2:

```
Reason:
   [ Image quality          ]
   [ Atypical morphology    ]
   [ Model false-positive   ]
   [ Model false-negative   ]
   [ Other                  ]
```

Tapping a reason completes the override. Audit entry written: `OVERRIDE_RECORDED` with `overrideContext = "live"`, `overrideReason = <selected>`, `actorId = <device clinician>`, `overrideActorInitials = null` (implicit), `overrideNotes = null`, `contextReviewed = null`.

Round-trip time: ~3 seconds. The microscopist returns to the camera ready for the next cell.

### Review override (deliberate, from History tab)

Triggered: user opens a flagged prediction in History and taps "Review and override."

Form (single screen, all fields visible):

```
The model said: PARASITIZED (53%)
Captured: 2026-03-14 09:23 UTC, session #...

Corrected verdict:
   [ Parasitized | Uninfected ]

Reason: (required)
   [ Image quality | Atypical morphology | ... ]

Override by: (defaults to device clinician's initials)
   [JM_______]

Notes: (optional)
   [_____________________________]

☐ I have reviewed the full session context for this prediction
                                              (required to enable Save)

[ Save override ]
```

The "I have reviewed" checkbox is required. The save button is disabled until it's checked, the verdict is chosen, and a reason is selected. The "Override by" defaults to the device clinician's initials but can be overwritten with different initials if a second person is reviewing.

Audit entry: `OVERRIDE_RECORDED` with `overrideContext = "review"`, all other fields populated, `contextReviewed = true`.

### What an override does to the prediction

The `Prediction.clinicianOverride` field is set to the corrected verdict. The original `parasitizedProb` and `uninfectedProb` are preserved unchanged. The display label shifts to show both ("Originally Parasitized 87%, overridden to Uninfected") in History and AI Analysis views.

An override cannot be undone in v1 (no edit history; once overridden, the prediction stays in that state). A subsequent override on the same prediction would create a new audit entry but the displayed verdict shows only the most recent override. This is a simplification — v1.1 could add edit history if needed.

### Override reason vocabulary — maintainer judgement

The five `OverrideReason` enum values are clinical concepts. Prior versions of this spec made their content gated on a clinical-advisor review before v1.0 release; that gate has been removed (see §24). The vocabulary now lands at maintainer discretion. The medical-device disclaimer in `NOTICE` continues to frame the application as research-only and not a basis for clinical diagnostic decisions; deployers seeking clinical validation under their own jurisdiction add it as part of their conformance work (§18).

### v1 single-clinician workaround

The override-time initials capture in v1 is a workaround for the single-clinician-device assumption (Section 9). v2 multi-clinician support would replace this with first-class actor switching: the active profile's `actorId` becomes the override actor directly. The `overrideActorInitials` field on `AuditEntry` would still exist for backwards compatibility with v1-era records, but new overrides under v2 would use the active profile's identity. This means v1 audit logs and v2 audit logs are interoperable — v2 readers ignore `overrideActorInitials` when it's null (always for v2-era rows) and use it when populated (always for v1-era rows).

---

## 13. Sessions and history

### Session formation

Implicit. The first prediction of any screening generates a new session UUID. Subsequent predictions within 30 minutes of the previous prediction inherit the same session UUID. A gap of 30+ minutes generates a new session UUID.

The 30-minute constant lives in `SessionGrouping.SESSION_GAP_MINUTES`. Not currently exposed in Settings; deployers needing a different value patch the shared code (which automatically affects both platforms). v1.1 may promote to a setting.

### Session labels

Free-text labels can be added retrospectively in the History tab → Sessions view → tap session → "Add label." The label is mirrored on every prediction in the session (denormalized for query simplicity) and displayed everywhere the session is referenced.

Constraints:
- Max 20 characters
- Plain ASCII only (rejects emoji and non-ASCII to keep export files clean)
- No validation against PII content — the clinician is responsible for following their clinic's privacy policy

A persistent warning copy appears on the label-editing UI: "Labels appear in exports and audit reports. Do not include identifying information per your clinic's privacy policy."

Audit: `SESSION_RELABELED` with the new label value in metadata.

### Duplicate marking

If a clinician realizes they scanned the same cell twice, they tap "Mark as duplicate of..." on the duplicate. The UI shows a picker of recent predictions (last 50 in the same session) to choose the original.

After marking:
- The duplicate gets `duplicateOfId = <original_id>`
- The duplicate is hidden from the default History list (filterable in by a "Show duplicates" toggle)
- Audit entry written: `PREDICTION_LINKED_AS_DUPLICATE`
- No data is deleted

### No hard delete in v1

A deployer that needs hard-delete adds it themselves and modifies their compliance documentation accordingly. The shared `Permissions` module has `MARK_AS_DUPLICATE` but not `DELETE_PREDICTION` — the v1 UI does not surface deletion at all on either platform.

---

## 14. Export

### Trigger

History tab → Data management → "Export all data." Requires a fresh biometric prompt regardless of session state.

### What gets exported

A single ZIP file containing one JSON document plus a README.

```
malaria-detector-export-{deviceUuid-prefix}-{timestamp}.zip
├── export.json
└── README.txt
```

The export format is byte-identical between platforms — same JSON keys, same ordering, same canonical English enum values. A bundle exported from iOS can be opened, inspected, and (in v1.1, when import lands) imported on Android, and vice versa.

### export.json schema

```json
{
  "schemaVersion": "1.0",
  "exportTimestamp": "ISO8601 UTC",
  "exportedByActorId": "UUID",
  "deviceUuid": "UUID",
  "platform": "ios" or "android",
  "clinicName": "string",
  "jurisdiction": "enum string",
  "lawfulBasis": "enum string",
  "appVersion": "semver",
  "osVersion": "string",
  
  "summary": {
    "predictionCount": "int",
    "sessionCount": "int",
    "auditEntryCount": "int",
    "consentRecordCount": "int",
    "firstPredictionAt": "ISO8601 UTC or null",
    "lastPredictionAt": "ISO8601 UTC or null"
  },
  
  "clinicianProfiles": [
    { "actorId": "UUID", "role": "enum string", "initials": "string or null",
      "enrolledAt": "ISO8601 UTC", "biometricEnrolled": "bool" }
  ],
  
  "predictions": [
    { "<every field from Prediction entity, ISO8601 timestamps>": "..." }
  ],
  
  "auditLog": [
    { "<every field from AuditEntry entity, ISO8601 timestamps>": "..." }
  ],
  
  "signature": "hex HMAC-SHA256 over all preceding fields"
}
```

### Signature

The export is signed with HMAC-SHA256. The key is derived deterministically from:
- The device UUID
- The timestamp salt embedded in the export

This makes the signature reproducible from the original device (for integrity verification by anyone with access to that device). It does not pretend to be an external notarization — for that, a deployer would add a clinic-supplied signing key in v2.

The signature is computed identically on both platforms (same HMAC-SHA256 implementation in the shared module via `org.kotlincrypto.hash`), so verification works regardless of which platform produced the bundle.

### Audit

- `EXPORT_INITIATED` when the user taps Export and authenticates
- `EXPORT_COMPLETED` when the file is generated, with the bundle size and signature recorded in metadata
- `EXPORT_FAILED` if any step errors

### Sharing

- **iOS:** native share sheet (`UIActivityViewController`) presents AirDrop, Files, Mail, etc.
- **Android:** native share intent (`Intent.ACTION_SEND`) presents Drive, Gmail, Files by Google, Nearby Share, etc.

### No import in v1

v1 produces bundles, does not consume them. Device migration in v1 means: export from old device, archive bundle in clinic's records, start fresh on new device. v1.1 may add import.

### What does NOT go in the export

- Any images (none exist)
- Device passcodes or biometric data
- The signature key itself
- Cached model files (reproducible from registry + Hugging Face)
- Crash logs (separate flow; see Section 16)

---

## 15. Localization

### Scope

**English only for v1.x.** The application ships English-only. Community localization is documented as a fork-extension point for deployers serving non-English contexts.

The repository carries dormant scaffolding from an earlier roadmap: `crowdin.yml` at the repo root, empty locale directories under `androidApp/src/main/res/values-{sw,fr,pt}/`, and `sw` / `fr` / `pt` entries inside `iosApp/Localization/Localizable.xcstrings`. This scaffolding is harmless future-proofing for a downstream deployer who chooses to revive translation under their own fork. The upstream maintainer is **not** soliciting translations and the Crowdin project is **not** provisioned.

Prior versions of this spec called for four ship-locales for v1 (English, Swahili, French, Portuguese). That roadmap item has been removed from the project scope (see §22 Phase 12 and §24).

### What stays English-only by design (independent of the cancellation)

Even if a deployer-fork revives translation, the following remain canonical English everywhere:

- Privacy policy, terms of service, decision-support disclaimer (full long-form legal text) — legal text translation requires professional review per jurisdiction.
- Onboarding compliance acknowledgements.
- Override reason display names (the five enum values render in English everywhere).
- Audit log action names in the viewer (the enum strings render in English).
- Any free-text content the user has entered.

Override reasons and audit log enums are stored in canonical English regardless of display locale (Section 5); translating their display strings would introduce ambiguity in audit-log analysis.

### Storage of the locale preference (still wired)

The onboarding language picker and Settings → Language continue to write a locale preference to `UserDefaults` (iOS) / `DataStore` (Android). Unencrypted because:
- Choice of language is not sensitive
- Must be available before authentication (the locked Settings tab needs to render in the user's language)
- Survives "Reset device" (the language stays; everything else resets)

Today this preference selects English only; a deployer-fork that revives translation can light up the other locale buckets without re-plumbing the preference machinery.

### RTL-readiness discipline (retained)

Section 11's RTL-readiness requirement stays in force. The discipline — semantic alignment (`.leading` / `.trailing` on SwiftUI, `start` / `end` on Compose), auto-mirrored directional icons, no absolute directional values — costs nothing during v1 development and remains valuable engineering practice even without ship-locale targets. A deployer-fork that ships an RTL language gets a layout that already flips correctly.

---

## 16. Crash logs

### Approach

On-device only, opt-in disclosure, no third-party analytics service. Same approach on both platforms; storage location and signal-handling mechanism differ.

When the app crashes, a structured crash log is written:
- **iOS:** `~/Documents/crashlogs/{incident-uuid}.json` with `NSFileProtectionComplete`
- **Android:** `context.filesDir/crashlogs/{incident-uuid}.json` with `EncryptedFile`

The user can view the count and share individual logs from Settings → Crash logs (biometric-gated).

### What goes in a crash log

- Stack trace
- App version, OS version, device model class (e.g., "iPhone15,2" / "Pixel 9 Pro" — model identifiers, not personally identifying)
- Last 50 audit log action types (action strings only — no resource IDs, no metadata, no actor IDs)
- Memory pressure at time of crash
- Whether device was locked / unlocked at time of crash
- Generated incident UUID

### What does NOT go in a crash log

- Any prediction data
- Any override notes or session labels
- Any clinician initials or actor UUIDs
- Any image data or hashes
- Any clinic configuration values (clinic name, jurisdiction, etc.)
- Any consent records

### Onboarding disclosure

A line during Phase 1 step 4 (medical disclaimer): "If the app crashes, a diagnostic log is saved on this device only. Nothing is sent automatically. You can review and share individual logs from Settings."

### Implementation constraints

The crash log writer must:
- Use only stack-allocated buffers where the language permits (no allocations during signal handling on iOS)
- Use direct file syscalls (no SwiftData / Room, no async, no Foundation/AndroidX)
- Avoid the audit log entirely (which uses persistence) — crashes during persistence shouldn't try to persist a record of the crash

Per-platform:
- **iOS:** `signal()` handler for native crashes; `Thread.setDefaultUncaughtExceptionHandler` for Swift errors. Writer uses POSIX `open()`, `write()`, `close()` only.
- **Android:** `Thread.setDefaultUncaughtExceptionHandler` for JVM crashes; native crashes captured via a small NDK signal handler. Writer uses `java.io.FileOutputStream` with pre-allocated buffers for the JVM path; native crashes use POSIX file APIs.

Crash logs are NOT in the database. They survive a database wipe. They auto-expire after 30 days (file modification time check on every app launch).

### Sharing

Settings → Crash logs shows a list with timestamp and incident UUID. Tapping a log opens the platform share sheet (iOS: `UIActivityViewController`; Android: `Intent.ACTION_SEND`). The user chooses the destination. Sharing is audited as `CRASH_LOG_SHARED` with the incident UUID.

---

## 17. What is explicitly NOT in v1

This section exists so a reader can quickly determine whether a missing feature is "we'll add it" or "we considered and excluded it." Items below are categorized by whether they're deferred to a planned version, deferred indefinitely (not on maintainer roadmap), or genuinely scope-excluded.

### Deferred to a planned version

- Import of export bundles (v1.1 — see Section 24)
- Explicit sessions with pseudonymous patient refs (v1.1 — see Section 24)
- Multi-clinician role separation on a single device (v2, design sketched in Section 24)

### Deferred indefinitely — not on maintainer roadmap (deployer-fork territory)

- **Localization to non-English locales.** The repository ships English-only. The Crowdin scaffolding (`crowdin.yml`, empty `values-{sw,fr,pt}/` directories, iOS xcstrings sw/fr/pt entries) remains in place as harmless future-proofing; a deployer who wishes to revive translation under their own fork can do so without re-plumbing the locale machinery. The upstream maintainer does not solicit translations and the Crowdin project is not provisioned. See Section 15 and Section 24.
- **Right-to-left language capability.** v1 is written RTL-ready (Section 11) as engineering discipline, but no RTL languages ship. A deployer-fork that adds an RTL translation lights up the capability. See Section 24.
- **Cloud-tier inference** (any server-side classification). Design reference in `docs/CLOUD_TIER_REFERENCE.md`. The `Classifier` interface accommodates a future `CloudClassifier` without architectural changes; the compliance burden (BAA, mTLS, region routing, key rotation) is deployer-owned.
- **Hard deletion of predictions** from the UI. Implementation pattern documented in `docs/COMPLIANCE.md`. A deployer with right-to-erasure obligations under their jurisdiction's law adds this to their fork.
- **App Store / Play Store distribution** by the maintainer. Generic release infrastructure with placeholder credentials exists; deployer fills in their own developer account details. See `docs/STORE_SUBMISSION.md`.
- **Research contribution of images for retraining datasets.** The image pipeline is in-memory only by design (Section 6). A deployer wanting research contribution implements it as an upload flow during the screening session, with consent capture appropriate to their jurisdiction and research-repository choice.

### Genuinely scope-excluded

- Image persistence beyond the screening session (images live in memory only, hashed for audit)
- Cryptographic chain hashing of audit log
- Per-jurisdiction auto-retention enforcement
- Active anomalous-access detection
- Edit history on overrides
- Demo video at v0.1 release
- Third-party crash analytics (Sentry, Crashlytics, etc.)
- Marketing site separate from the GitHub README

### v2-conditional (evidence-gated)

- **Compose Multiplatform** UI sharing. The v1 native-UI decision stands. If dual-UI maintenance burden proves unsustainable based on real v1.x deployment evidence, reconsideration in v2 is permitted for specific screens. The shared-module/native-UI boundary itself is fixed; the only question would be whether some screens within the native-UI layer migrate.

Each item has reasoning in the prior sections, in Section 24's open-questions, or in the build-plan section.

---

## 18. Compliance posture

### Framing

This is a research-prototype open-source application licensed under Hippocratic 3.0 with explicit medical-device disclaimer. The maintainer does not pursue regulatory clearance. A deployer who wants to use it in a clinical setting takes on the conformance burden for their jurisdiction.

The application is built so that the deployer's compliance work is *additive*, not corrective. Encryption at rest is implemented on both platforms. Audit logging is implemented on both platforms. Biometric gate is implemented on both platforms. The architecture supports the compliance features a deployer needs to add (chain hashing, auto-retention, anomalous-access detection, mTLS for cloud tier).

### What v1 implements (on both platforms)

- At-rest encryption — `NSFileProtectionComplete` (iOS) and SQLCipher + Keystore-managed key (Android)
- Hardware-backed encryption keys where the platform supports it (Secure Enclave on iOS, StrongBox on Android when available, software-backed keystore otherwise)
- Biometric/passcode gate on app launch
- Auto-logout after configurable inactivity
- Complete audit log with structured action vocabulary
- No image persistence
- Pseudonymous clinician identity
- Override flow with attribution to the device clinician (and free-text initials capture for live multi-actor scenarios)
- Decision-support framing throughout (every prediction is overridable; the UI never presents a verdict as final)
- Lawful-basis capture at onboarding
- Per-jurisdiction retention policy *displayed* in Settings
- Onboarding consent acknowledgements with timestamped audit records
- Signed export bundles
- Privacy-preserving crash logs (no PHI, no analytics service)

### What v1 explicitly does not implement

These are documented in `COMPLIANCE.md` as deployer responsibilities:

- DPIA (Data Protection Impact Assessment) under GDPR Art. 35
- BAA (Business Associate Agreement) with any third-party service
- Cryptographic chain hashing of audit log with daily integrity verification
- Auto-enforcement of retention policy (delete records after N years)
- Anomalous-access detection
- Penetration testing
- App Store / Play Store medical-app submission
- Notified Body conformity assessment (EU MDR)
- FDA 510(k) submission (US)
- ISO 13485 quality management system
- IEC 62304 medical device software lifecycle compliance
- Privacy policy and terms of service review by qualified counsel per jurisdiction
- Clinical validation studies

A deployer wanting clinical use takes on the items above as part of their own conformance work. The maintainer provides the application; the deployer provides the regulatory posture.

### GDPR Art. 22 (no autonomous decisions)

Every prediction in the app is overridable. The UI never presents a verdict as final. The decision-support disclaimer is shown at onboarding and linked from About at all times. This implements the regulatory expectation that AI predictions affecting natural persons remain subject to human review. A deployer in the EU may need additional clinical workflow documentation; the app itself complies with the architectural expectation.

### Platform-specific compliance surface

**iOS-specific:**
- App Sandbox (default in iOS)
- App Transport Security: TLS 1.3 mandatory for cloud connections; ATS exception entries forbidden except for `huggingface.co`
- Privacy Manifest (`PrivacyInfo.xcprivacy`) declaring health data type, no tracking
- Privacy Nutrition Label declaring Health & Fitness category, no tracking

**Android-specific:**
- Scoped storage (default in API 36)
- Network Security Configuration: TLS pinning for cloud connections (informational in v1 since no cloud tier); cleartext traffic blocked
- Data Safety section in Play Store listing declaring Health/Fitness, encryption in transit and at rest, retention policy
- `allowBackup="false"` and explicit data extraction rules preventing automatic cloud/device backup

### Documentation deliverables in v1

- `LICENSE` — Hippocratic 3.0 text
- `NOTICE` — medical-device disclaimer
- `docs/COMPLIANCE.md` — what's implemented vs deferred, per platform
- `docs/SCHEMA.md` — canonical persistence schema
- `SECURITY.md` — private disclosure procedure for security issues

---

## 19. Build configuration

### Top-level Gradle

```kotlin
plugins {
    kotlin("multiplatform") version "2.1.0" apply false
    kotlin("plugin.serialization") version "2.1.0" apply false
    id("com.android.application") version "8.7.3" apply false
    id("com.android.library") version "8.7.3" apply false
    id("androidx.room") version "2.7.0" apply false
}
```

### Shared module

The shared module produces:
- An Android AAR consumed via `implementation(project(":shared"))` in `androidApp`
- An iOS XCFramework consumed via Swift Package Manager in the Xcode project

```kotlin
// shared/build.gradle.kts

plugins {
    kotlin("multiplatform")
    kotlin("plugin.serialization")
    id("com.android.library")
}

kotlin {
    androidTarget()

    val xcframeworkName = "Shared"
    val xcf = XCFramework(xcframeworkName)

    listOf(
        iosX64(),
        iosArm64(),
        iosSimulatorArm64()
    ).forEach {
        it.binaries.framework {
            baseName = xcframeworkName
            isStatic = true
            xcf.add(this)
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.1")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
                implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.6.1")
                implementation("com.benasher44:uuid:0.8.4")
                implementation("org.kotlincrypto.hash:sha2:0.5.6")
                // Note: no Ktor, no SQLDelight, no cloud-tier libraries
                // Persistence is platform-native; cloud tier is deferred
            }
        }
        val iosMain by getting {
            dependencies {
                // Core ML, Vision, AVFoundation, LocalAuthentication, SwiftData via cinterop
            }
        }
        val androidMain by getting {
            dependencies {
                // LiteRT and platform APIs are dependencies of androidApp, not shared
                // (shared androidMain just provides actual classes that interface with them)
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
            }
        }
    }
}

android {
    namespace = "com.malaria.shared"
    compileSdk = 36
    defaultConfig {
        minSdk = 36
    }
}

// Custom task to build the XCFramework for distribution
tasks.register("assembleSharedXCFramework") {
    dependsOn("assembleSharedReleaseXCFramework")
}
```

### iOS app — Swift Package Manager integration

The iOS app references the shared XCFramework via a local Swift Package. The package is generated as part of the build and lives at `iosApp/SharedFramework/`.

```swift
// iosApp/SharedFramework/Package.swift

// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Shared",
    platforms: [.iOS(.v26)],
    products: [
        .library(name: "Shared", targets: ["Shared"])
    ],
    targets: [
        .binaryTarget(
            name: "Shared",
            path: "Shared.xcframework"
        )
    ]
)
```

The Xcode project adds this package as a local Swift Package dependency. The `assembleSharedXCFramework` Gradle task copies the produced `Shared.xcframework` into `iosApp/SharedFramework/` so that Xcode picks it up.

Build flow:
1. Developer (or CI) runs `./gradlew :shared:assembleSharedXCFramework`
2. Gradle produces `shared/build/XCFrameworks/release/Shared.xcframework`
3. A post-build script (or `pre-actions` in the Xcode scheme) copies it to `iosApp/SharedFramework/Shared.xcframework`
4. Xcode build picks up the updated package

For development convenience, the Xcode scheme has a "Run Script" pre-action that invokes the Gradle task automatically when the iOS scheme is built. This means developers can hit Cmd+R in Xcode and get the latest shared code without needing to remember the Gradle step.

### iOS app — Info.plist (critical entries)

```xml
<key>NSCameraUsageDescription</key>
<string>Used to capture cell images from the microscope eyepiece for malaria screening.</string>

<key>NSFaceIDUsageDescription</key>
<string>Used to unlock the app and protect medical data.</string>

<key>NSAppTransportSecurity</key>
<dict>
  <key>NSAllowsArbitraryLoads</key><false/>
  <key>NSExceptionDomains</key>
  <dict>
    <key>huggingface.co</key>
    <dict>
      <key>NSExceptionMinimumTLSVersion</key><string>TLSv1.3</string>
      <key>NSExceptionRequiresForwardSecrecy</key><true/>
    </dict>
  </dict>
</dict>
```

### iOS app — PrivacyInfo.xcprivacy

```xml
<plist version="1.0">
<dict>
  <key>NSPrivacyCollectedDataTypes</key>
  <array>
    <dict>
      <key>NSPrivacyCollectedDataType</key>
      <string>NSPrivacyCollectedDataTypeHealth</string>
      <key>NSPrivacyCollectedDataTypeLinked</key><false/>
      <key>NSPrivacyCollectedDataTypeTracking</key><false/>
      <key>NSPrivacyCollectedDataTypePurposes</key>
      <array>
        <string>NSPrivacyCollectedDataTypePurposeAppFunctionality</string>
      </array>
    </dict>
  </array>
  <key>NSPrivacyTracking</key><false/>
  <key>NSPrivacyTrackingDomains</key><array/>
</dict>
</plist>
```

### iOS app — SwiftData container setup

```swift
// iosApp/Persistence/ModelContainerFactory.swift

enum ModelContainerFactory {
    static func make() throws -> ModelContainer {
        let schema = Schema([
            Prediction.self,
            AuditEntry.self,
            ClinicianProfile.self,
            ConsentRecord.self,
        ])

        let storeURL = try FileManager.default
            .url(for: .applicationSupportDirectory, in: .userDomainMask,
                 appropriateFor: nil, create: true)
            .appendingPathComponent("Malaria.store")

        try? FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.complete],
            ofItemAtPath: storeURL.path
        )

        let config = ModelConfiguration(
            schema: schema,
            url: storeURL,
            cloudKitDatabase: .none  // medical data MUST NOT sync to iCloud
        )

        return try ModelContainer(for: schema, configurations: [config])
    }
}
```

### iOS app — Environment keys for service injection

```swift
// iosApp/Environment/EnvironmentKeys.swift

import SwiftUI
import Shared  // KMP framework

// Each service has its own EnvironmentKey. defaultValue crashes
// to make missing wiring fail loudly in development.

private struct ClassifierKey: EnvironmentKey {
    static let defaultValue: Classifier = {
        fatalError("Classifier not provided. Set .environment(\\.classifier, ...) at app root.")
    }()
}

private struct AuthGateKey: EnvironmentKey {
    static let defaultValue: AuthGate = {
        fatalError("AuthGate not provided. Set .environment(\\.authGate, ...) at app root.")
    }()
}

private struct CameraServiceKey: EnvironmentKey {
    static let defaultValue: CameraService = {
        fatalError("CameraService not provided.")
    }()
}

// Repeat for: ModelRegistryService, PredictionStore, AuditLog, OnboardingState

extension EnvironmentValues {
    var classifier: Classifier {
        get { self[ClassifierKey.self] }
        set { self[ClassifierKey.self] = newValue }
    }
    var authGate: AuthGate {
        get { self[AuthGateKey.self] }
        set { self[AuthGateKey.self] = newValue }
    }
    var cameraService: CameraService {
        get { self[CameraServiceKey.self] }
        set { self[CameraServiceKey.self] = newValue }
    }
    // ... etc
}
```

### iOS app — Composition root

```swift
// iosApp/MalariaDetectorApp.swift

import SwiftUI
import SwiftData
import Shared

@main
struct MalariaDetectorApp: App {

    // Services are stored as @State so they live for the app's lifetime
    // and survive scene transitions. SwiftUI 6 + Swift 6.1: @Observable
    // services are Sendable; @State holds them with proper isolation.

    @State private var classifier: Classifier
    @State private var authGate: AuthGate
    @State private var cameraService: CameraService
    @State private var modelRegistry: ModelRegistryService
    @State private var predictionStore: PredictionStore
    @State private var auditLog: AuditLog
    @State private var onboardingState: OnboardingState

    let modelContainer: ModelContainer

    init() {
        // ModelContainer must be set up before services that depend on it
        let container = try! ModelContainerFactory.make()
        self.modelContainer = container

        // Construct all services once. They live for app lifetime.
        let context = ModelContext(container)
        _classifier = State(initialValue: CoreMLClassifier(modelId: "BNLeaky_Keras"))
        _authGate = State(initialValue: AuthGate(context: context))
        _cameraService = State(initialValue: CameraService())
        _modelRegistry = State(initialValue: ModelRegistryService())
        _predictionStore = State(initialValue: PredictionStore(context: context))
        _auditLog = State(initialValue: AuditLog(context: context))
        _onboardingState = State(initialValue: OnboardingState(context: context))
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .modelContainer(modelContainer)
                .environment(\.classifier, classifier)
                .environment(\.authGate, authGate)
                .environment(\.cameraService, cameraService)
                .environment(\.modelRegistry, modelRegistry)
                .environment(\.predictionStore, predictionStore)
                .environment(\.auditLog, auditLog)
                .environment(\.onboardingState, onboardingState)
        }
    }
}
```

### iOS app — Example @Observable service and view consumption

```swift
// iosApp/Services/AuthGate.swift

import Foundation
import LocalAuthentication
import Observation

@Observable
@MainActor
final class AuthGate {

    enum State: Sendable {
        case locked
        case unlocked(sessionStart: Date)
        case provisionedUnclaimed
    }

    private(set) var state: State = .locked

    private let context: ModelContext
    private let timer: SessionTimer

    init(context: ModelContext, timeoutMinutes: Int = 30) {
        self.context = context
        self.timer = SessionTimer(timeoutMinutes: Int32(timeoutMinutes)) { [weak self] in
            await self?.lockSession(reason: .timeout)
        }
    }

    func unlock() async throws {
        let laContext = LAContext()
        var error: NSError?
        guard laContext.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error) else {
            throw AuthError.noBiometricsConfigured
        }
        let success = try await laContext.evaluatePolicy(
            .deviceOwnerAuthentication,
            localizedReason: "Unlock Malaria Detector"
        )
        guard success else { throw AuthError.userCancelled }
        state = .unlocked(sessionStart: Date())
        await writeAudit(.sessionUnlocked)
    }

    func lockSession(reason: LockReason) async {
        state = .locked
        await writeAudit(.sessionRelocked(reason: reason))
    }

    func touch() { timer.touch() }
}
```

```swift
// iosApp/Views/HomeTab.swift

import SwiftUI
import SwiftData

struct HomeTab: View {

    @Environment(\.classifier) private var classifier
    @Environment(\.authGate) private var authGate
    @Environment(\.cameraService) private var cameraService
    @Environment(\.predictionStore) private var predictionStore

    // Direct SwiftData query — no ViewModel intermediate.
    // @Query is reactive; the view re-renders when new predictions arrive.
    @Query(sort: \Prediction.timestamp, order: .reverse)
    private var recentPredictions: [Prediction]

    var body: some View {
        switch authGate.state {
        case .locked:
            LockedPlaceholder()
        case .unlocked:
            ActiveScreeningView(recentPrediction: recentPredictions.first)
        case .provisionedUnclaimed:
            ProvisioningIncompleteView()
        }
    }
}
```

### Android app — Gradle module

```kotlin
// androidApp/build.gradle.kts

plugins {
    id("com.android.application")
    kotlin("android")
    kotlin("plugin.compose")
    id("androidx.room")
    id("kotlin-kapt")
}

android {
    namespace = "com.malaria.android"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.malaria.detector"
        minSdk = 36
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    bundle {
        density { enableSplit = false }
        abi { enableSplit = false }
    }

    packaging {
        resources.excludes.add("META-INF/*")
    }

    room {
        schemaDirectory("$projectDir/schemas")  // for migration testing
    }
}

dependencies {
    implementation(project(":shared"))

    // Compose BOM (April 2026 release)
    implementation(platform("androidx.compose:compose-bom:2026.04.01"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.10.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.0")  // collectAsStateWithLifecycle
    implementation("androidx.navigation:navigation-compose:2.9.0")
    // Note: no lifecycle-viewmodel-compose; the app does not use ViewModels (Section 4)

    // Room with SQLCipher
    implementation("androidx.room:room-runtime:2.7.0")
    implementation("androidx.room:room-ktx:2.7.0")
    kapt("androidx.room:room-compiler:2.7.0")
    implementation("net.zetetic:sqlcipher-android:4.6.1@aar")
    implementation("androidx.sqlite:sqlite-ktx:2.4.0")

    // ML inference (LiteRT)
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("com.google.ai.edge.litert:litert:1.0.1")

    // Camera (CameraX)
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")

    // Security and biometrics
    implementation("androidx.biometric:biometric:1.2.0-alpha05")
    implementation("androidx.security:security-crypto-ktx:1.1.0-alpha06")

    // Preferences (unencrypted; for language only)
    implementation("androidx.datastore:datastore-preferences:1.1.1")

    // Networking (for Hugging Face model downloads)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("androidx.work:work-runtime-ktx:2.10.0")  // resumable downloads

    // Test
    testImplementation("junit:junit:4.13.2")
    testImplementation("androidx.room:room-testing:2.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
```

### Android app — AndroidManifest.xml

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
          xmlns:tools="http://schemas.android.com/tools">

  <uses-permission android:name="android.permission.CAMERA" />
  <uses-permission android:name="android.permission.USE_BIOMETRIC" />
  <uses-permission android:name="android.permission.INTERNET" />

  <uses-feature android:name="android.hardware.camera" android:required="true" />
  <uses-feature android:name="android.hardware.camera.autofocus" android:required="true" />

  <application
      android:name=".MalariaApplication"
      android:label="@string/app_name"
      android:icon="@mipmap/ic_launcher"
      android:roundIcon="@mipmap/ic_launcher_round"
      android:allowBackup="false"
      android:fullBackupContent="false"
      android:dataExtractionRules="@xml/data_extraction_rules"
      android:networkSecurityConfig="@xml/network_security_config"
      android:theme="@style/Theme.MalariaDetector"
      tools:targetApi="36">

    <activity
        android:name=".MainActivity"
        android:exported="true"
        android:screenOrientation="portrait"
        android:theme="@style/Theme.MalariaDetector">
      <intent-filter>
        <action android:name="android.intent.action.MAIN" />
        <category android:name="android.intent.category.LAUNCHER" />
      </intent-filter>
    </activity>

  </application>
</manifest>
```

### Android app — Room database setup

```kotlin
// androidApp/src/main/kotlin/com/malaria/android/data/MalariaDatabase.kt

@Database(
    entities = [Prediction::class, AuditEntry::class, ClinicianProfile::class, ConsentRecord::class],
    version = 1,
    exportSchema = true
)
@TypeConverters(InstantConverter::class)
abstract class MalariaDatabase : RoomDatabase() {
    abstract fun predictionDao(): PredictionDao
    abstract fun auditDao(): AuditDao
    abstract fun clinicianDao(): ClinicianDao
    abstract fun consentDao(): ConsentDao

    companion object {
        fun create(context: Context): MalariaDatabase {
            val passphrase = SecureKeyStore.getOrCreateDatabaseKey(context)
            val factory = SupportFactory(passphrase)

            return Room.databaseBuilder(context, MalariaDatabase::class.java, "malaria.db")
                .openHelperFactory(factory)
                .fallbackToDestructiveMigration(dropAllTables = false)  // explicit for v1
                .build()
        }
    }
}

// Hardware-backed key from Android Keystore, with StrongBox preference

object SecureKeyStore {
    private const val KEY_ALIAS = "malaria_db_key_v1"

    fun getOrCreateDatabaseKey(context: Context): ByteArray {
        val keystore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        if (!keystore.containsAlias(KEY_ALIAS)) {
            val builder = KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .setUserAuthenticationRequired(true)
                .setUserAuthenticationParameters(300, KeyProperties.AUTH_BIOMETRIC_STRONG or KeyProperties.AUTH_DEVICE_CREDENTIAL)

            // StrongBox if available, software-backed fallback
            if (context.packageManager.hasSystemFeature(PackageManager.FEATURE_STRONGBOX_KEYSTORE)) {
                builder.setIsStrongBoxBacked(true)
            }

            KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
                .apply { init(builder.build()) }
                .generateKey()
        }
        return deriveSqlCipherKey(KEY_ALIAS)  // derives byte[] passphrase for SQLCipher
    }
}
```

### Android app — CompositionLocal keys for service injection

```kotlin
// androidApp/src/main/kotlin/com/malaria/android/ui/locals/AppLocals.kt

package com.malaria.android.ui.locals

import androidx.compose.runtime.compositionLocalOf
import com.malaria.android.services.AuthGate
import com.malaria.android.services.CameraService
import com.malaria.android.services.ClassifierService
import com.malaria.android.services.PredictionStore
import com.malaria.android.services.AuditLog
import com.malaria.android.services.ModelRegistryService
import com.malaria.android.services.OnboardingState

// Each service has its own CompositionLocal. The default lambda
// throws to make missing wiring fail loudly in development.

val LocalClassifier = compositionLocalOf<ClassifierService> {
    error("LocalClassifier not provided. Wrap in CompositionLocalProvider at root.")
}

val LocalAuthGate = compositionLocalOf<AuthGate> {
    error("LocalAuthGate not provided.")
}

val LocalCameraService = compositionLocalOf<CameraService> {
    error("LocalCameraService not provided.")
}

val LocalModelRegistry = compositionLocalOf<ModelRegistryService> {
    error("LocalModelRegistry not provided.")
}

val LocalPredictionStore = compositionLocalOf<PredictionStore> {
    error("LocalPredictionStore not provided.")
}

val LocalAuditLog = compositionLocalOf<AuditLog> {
    error("LocalAuditLog not provided.")
}

val LocalOnboardingState = compositionLocalOf<OnboardingState> {
    error("LocalOnboardingState not provided.")
}
```

### Android app — Composition root

```kotlin
// androidApp/src/main/kotlin/com/malaria/android/MainActivity.kt

package com.malaria.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import com.malaria.android.data.MalariaDatabase
import com.malaria.android.services.*
import com.malaria.android.ui.RootScreen
import com.malaria.android.ui.locals.*
import com.malaria.android.ui.theme.MalariaTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class MainActivity : ComponentActivity() {

    // Services constructed once. They live for the activity's lifetime,
    // which exceeds any individual composition's lifetime.

    private val database by lazy { MalariaDatabase.create(applicationContext) }
    private val uiScope by lazy { CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate) }
    private val computeScope by lazy { CoroutineScope(SupervisorJob() + Dispatchers.Default) }

    private val classifier by lazy {
        ClassifierService(modelId = "BNLeaky_Keras", scope = computeScope)
    }
    private val authGate by lazy {
        AuthGate(database = database, scope = uiScope)
    }
    private val cameraService by lazy {
        CameraService(scope = computeScope)
    }
    private val modelRegistry by lazy {
        ModelRegistryService(scope = computeScope)
    }
    private val predictionStore by lazy {
        PredictionStore(database = database, scope = uiScope)
    }
    private val auditLog by lazy {
        AuditLog(database = database, scope = uiScope)
    }
    private val onboardingState by lazy {
        OnboardingState(database = database, scope = uiScope)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MalariaTheme {
                CompositionLocalProvider(
                    LocalClassifier provides classifier,
                    LocalAuthGate provides authGate,
                    LocalCameraService provides cameraService,
                    LocalModelRegistry provides modelRegistry,
                    LocalPredictionStore provides predictionStore,
                    LocalAuditLog provides auditLog,
                    LocalOnboardingState provides onboardingState,
                ) {
                    RootScreen()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        uiScope.coroutineContext[kotlinx.coroutines.Job]?.cancel()
        computeScope.coroutineContext[kotlinx.coroutines.Job]?.cancel()
    }
}
```

### Android app — Example service and composable consumption

```kotlin
// androidApp/src/main/kotlin/com/malaria/android/services/AuthGate.kt

package com.malaria.android.services

import com.malaria.android.data.MalariaDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant

class AuthGate(
    private val database: MalariaDatabase,
    private val scope: CoroutineScope,
    timeoutMinutes: Int = 30,
) {
    sealed interface State {
        data object Locked : State
        data class Unlocked(val sessionStart: Instant) : State
        data object ProvisionedUnclaimed : State
    }

    private val _state = MutableStateFlow<State>(State.Locked)
    val state: StateFlow<State> = _state.asStateFlow()

    // BiometricPrompt-driven unlock invoked from the Activity scope.
    // The actual prompt is shown by the consuming composable via a helper
    // that bridges to BiometricPrompt's callback API.
    suspend fun unlock(onPromptRequired: suspend () -> Boolean) {
        val success = onPromptRequired()
        if (success) {
            _state.value = State.Unlocked(Clock.System.now())
            scope.launch { database.auditDao().writeSessionUnlocked() }
        }
    }

    fun lockSession(reason: LockReason) {
        _state.value = State.Locked
        scope.launch { database.auditDao().writeSessionRelocked(reason) }
    }
}
```

```kotlin
// androidApp/src/main/kotlin/com/malaria/android/ui/screens/HomeScreen.kt

package com.malaria.android.ui.screens

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.services.AuthGate
import com.malaria.android.ui.locals.LocalAuthGate
import com.malaria.android.ui.locals.LocalPredictionStore

@Composable
fun HomeScreen() {
    val authGate = LocalAuthGate.current
    val predictionStore = LocalPredictionStore.current

    // Direct service consumption — no ViewModel intermediate.
    val authState by authGate.state.collectAsStateWithLifecycle()
    val recentPredictions by predictionStore.recent.collectAsStateWithLifecycle()

    when (authState) {
        is AuthGate.State.Locked -> LockedPlaceholder()
        is AuthGate.State.Unlocked -> ActiveScreeningView(
            recentPrediction = recentPredictions.firstOrNull()
        )
        is AuthGate.State.ProvisionedUnclaimed -> ProvisioningIncompleteView()
    }
}
```

### Android — network_security_config.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
  <base-config cleartextTrafficPermitted="false">
    <trust-anchors>
      <certificates src="system" />
    </trust-anchors>
  </base-config>
  <domain-config>
    <domain includeSubdomains="true">huggingface.co</domain>
    <pin-set expiration="2027-01-01">
      <pin digest="SHA-256">REPLACE_WITH_ACTUAL_PIN</pin>
      <pin digest="SHA-256">REPLACE_WITH_BACKUP_PIN</pin>
    </pin-set>
  </domain-config>
</network-security-config>
```

### Android — data_extraction_rules.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<data-extraction-rules>
  <cloud-backup><exclude domain="root" /></cloud-backup>
  <device-transfer><exclude domain="root" /></device-transfer>
</data-extraction-rules>
```

Combined with `allowBackup="false"` in the manifest, this means medical data cannot leave the device via Android Auto Backup, Google Drive backup, or device-to-device transfer.

---

## 20. Testing strategy

### Shared module (commonTest)

- Unit tests for `Threshold` — boundary cases, default values
- Unit tests for `SessionGrouping` — single prediction, within-gap, across-gap, edge cases at exactly 30 minutes
- Unit tests for `Permissions` — every (role, action) combination
- Unit tests for `RetentionPolicy` — every jurisdiction
- Unit tests for `ModelRegistry` — parsing valid JSON, malformed JSON, missing fields, version-mismatch handling

These run on both iOS (via XCTest under the hood when run via Gradle) and JVM. Both must pass on every PR.

### iOS-specific tests (XCTest)

- SwiftData CRUD on each entity type
- `NSFileProtectionComplete` is actually applied (file-system attribute check)
- DTO → SwiftData entity mapping is lossless and idempotent
- `AuthGate` (and other `@Observable` services) state transitions verified
- Service composition root produces a fully-wired environment (no `fatalError` from missing `EnvironmentKey` access)
- Crash log writer is signal-safe (manual test plus a controlled-crash test target)

SwiftUI snapshot tests for each screen in each language. Locked state and unlocked state. Tests provide test-double services via `.environment(\.classifier, MockClassifier())` etc. — the no-ViewModels architecture means tests construct environments directly, no ViewModel mocking required.

### Android-specific tests

- Unit tests for services (`AuthGate`, `CameraService`, `ClassifierService`, `PredictionStore`, `AuditLog`) and repositories — pure JVM where possible, Robolectric only where Android APIs are unavoidable
- Service state transitions verified via `StateFlow` test collectors (`turbine` library or hand-rolled `Flow` assertions)
- Room database tests on Android instrumentation (`androidTest`) — CRUD on each entity, encryption verified by attempting to read the DB file directly
- DTO → Room entity mapping is lossless and idempotent
- Compose UI tests for each screen using `androidx.compose.ui.test`. Tests provide test-double services via `CompositionLocalProvider` — same pattern as production composition root, just with fakes.

Compose preview tests for each screen in each language. Locked state and unlocked state.

### Integration tests (both platforms)

End-to-end inference: load the bundled model, classify a fixed set of test images, verify probabilities match expected values within float tolerance (the same fixed test set used in the notebook's Part 2 evaluation). This catches:
- Core ML / TFLite conversion drift between toolchain versions
- iOS / Android runtime behavior changes
- Preprocessing implementation drift

Test images are checked into the repo at `iosApp/Tests/Resources/` and `androidApp/src/androidTest/assets/test_images/` (small, public, dataset-attributed).

### Schema-drift tests

A CI test on both platforms compares the actual entity definitions against `docs/SCHEMA.md`:
- iOS: SwiftData `@Model` reflection dumps the inferred schema; compared against a serialized form of `SCHEMA.md`
- Android: Room's `@Database(exportSchema = true)` produces a JSON schema export; compared against `SCHEMA.md`

Any drift fails the build on the affected platform.

These CI tests are the **official mitigation** for schema drift between SwiftData and Room. A schema-first code generator that would emit both `@Model` and `@Entity` from a single source was considered and rejected — the 4-entity schema is small enough that the marginal cost of updating two files per schema change is lower than the cost of maintaining a code generator. `docs/SCHEMA.md` documents the conditions under which this trade-off should be revisited (substantial schema growth or frequent migrations).

### Manual test plan

A `docs/MANUAL_TEST_PLAN.md` covers what cannot be automated, on both platforms:
- Biometric prompt actually appears in real device contexts
- Camera permission flow
- Background-resume after auto-logout
- Hugging Face download with intermittent connectivity
- Onboarding flow on a fresh device
- Reset device flow
- **Force RTL verification:** every screen rendered with the platform's "Force RTL" developer option enabled, even with English text. Verifies that no LTR-only assumptions leaked into the layout. Required before v0.1 ships even though no RTL languages are active in v1.x; the discipline ensures a deployer-fork that adds an RTL translation gets a working layout without re-implementation.

### Continuous real-device sanity-checking

The Phase 15 manual test plan is the formal end-of-build verification, but real-device sanity-checking happens continuously from Phase 5 onward. Specifically:

- **Phase 5 (camera and live screening):** every camera-related change is sanity-checked on a real iPhone before merging. AVFoundation's simulator behavior diverges enough from device behavior that simulator-only verification gives false confidence.
- **Phase 6 (override flow), Phase 7 (history), Phase 8 (settings):** any change touching biometric flows is verified on real iPhone — `LAContext` semantics differ between simulator and device in failure-and-fallback scenarios.
- **Phase 11 (export, crash logs):** share-sheet integration and file-protection attributes are verified on real iPhone.

For Android, equivalent continuous checking on a Pixel emulator covers most cases; real Pixel hardware is preferred for biometric and camera work but not required for v0.1 (per the maintainer's hardware reality acknowledged elsewhere).

This continuous-checking discipline catches platform-specific bugs while they're cheap to fix, rather than discovering them en masse at Phase 15.

### Coverage target

Shared module: ≥90% line coverage (enforced in CI).
iOS app: ≥70% line coverage for non-UI code; UI covered by snapshot tests.
Android app: ≥70% line coverage for non-UI code; UI covered by Compose preview tests.

---

## 21. CI/CD

### Workflows

`.github/workflows/ci-shared.yml`:
- Triggers on every PR touching `shared/`
- Runs on macOS (for iOS targets) and Linux (for JVM targets)
- Executes `./gradlew :shared:check`
- Reports coverage; fails if below threshold

`.github/workflows/ci-ios.yml`:
- Triggers on every PR touching `iosApp/` or `shared/`
- Runs on macOS
- Builds the iOS app for simulator
- Runs XCTest suite
- Runs SwiftLint
- Validates `PrivacyInfo.xcprivacy` via `xcrun privacycli`
- Validates SwiftData schema against `SCHEMA.md`

`.github/workflows/ci-android.yml`:
- Triggers on every PR touching `androidApp/` or `shared/`
- Runs on Linux
- Builds the Android app
- Runs JVM unit tests via `./gradlew :androidApp:test`
- Runs Android instrumentation tests on an emulator (`reactivecircus/android-emulator-runner`)
- Runs ktlint and Android Lint
- Validates Room schema export against `SCHEMA.md`

`.github/workflows/release-ios.yml`:
- Triggers on tag push matching `v*`
- Builds and archives iOS app
- Uploads to TestFlight (requires Apple Developer Program enrollment by deployer or maintainer)

`.github/workflows/release-android.yml`:
- Triggers on tag push matching `v*`
- Builds signed AAB
- Uploads to Play Internal Testing track (requires Play Console enrollment)

### Branch protection

- `main` requires: passing CI on all four workflows (shared, iOS, Android, plus the relevant release workflow if a tag is pushed), ≥1 maintainer review for code changes
- Force-push disabled on `main`
- Signed commits required for releases

### Secrets

- App Store Connect API key
- Play Console service account JSON
- Hugging Face token (for release pipeline that uploads new model versions)
- Apple notarization credentials (for distributing outside App Store)

All scoped to specific workflows; none accessible to forks.

---

## 22. Phased build plan

The build plan interleaves both platforms. Each phase has a deliverable for both iOS and Android. The shared module is built once at the start and consumed by both platforms throughout.

### Phase -1 — Prerequisites (1 week, before code-writing begins)

Two artifacts must exist before Phase 0 starts:

1. **Notebook outputs reproducible.** Re-run the ML notebook's Part 7 (Core ML export pipeline) and confirm it produces `.mlpackage` files for all 18 models identical to what's currently in hand. This catches any drift between the notebook code and the existing exports — if the notebook produces different outputs today than it did originally, the differences need to be understood before they're bundled into the apps.

2. **Notebook Part 7B: TFLite export pipeline.** Extend the notebook with a parallel export step generating `.tflite` files for all 18 models alongside the existing Core ML exports. The work is 1–2 days: load each model, convert via TFLite Converter (with int8 quantization for the smaller models, float16 for the larger ones), save with appropriate metadata, verify each `.tflite` output produces inference results within float-tolerance of the matching `.mlpackage` on the same test images.

Outputs of Phase -1:
- `Malaria_BNLeaky_Keras.mlpackage` and `Malaria_BNLeaky_Keras.tflite` (the bundled model — both platforms)
- 17 additional `.mlpackage` files (iOS-only models, for Hugging Face upload)
- 17 additional `.tflite` files (Android-only models, for Hugging Face upload)
- SHA-256 hashes recorded for all 36 model files

This work belongs in the notebook, not in the app codebase. It's pre-Phase-0 work that establishes the model-artifact foundation for everything else.

### Phase 0 — Project bootstrap (2 weeks)

- Create KMP project structure
- Configure Gradle for shared module producing both XCFramework and AAR
- Set up Xcode project consuming shared via SPM (local Swift Package)
- Set up Android Gradle module consuming shared
- Verify cross-platform build: a dummy `expect class Greeter` works from both SwiftUI and Compose
- Set up GitHub repo (private until v0.1)
- Set up Hugging Face repo for model artifacts
- License, NOTICE, README scaffolding
- All four CI workflows operational with skeleton tests passing

### Phase 1 — Shared module foundation (2 weeks)

- Implement all commonMain types from Section 5
- Threshold, SessionGrouping, Permissions, RetentionPolicy
- ModelRegistry parsing
- Domain DTOs and enums
- Image preprocessing
- `Classifier` expect class
- Full commonTest coverage at the target threshold

### Phase 2 — iOS inference plumbing (1.5 weeks)

- `CoreMLClassifier` actual class in iosMain
- Bundle `Malaria_BNLeaky_Keras.mlpackage`
- Throwaway test harness: load image from photo library → classify → print probabilities
- End-to-end inference test against fixed test images

### Phase 3 — Android inference plumbing (1.5 weeks)

- `TFLiteClassifier` actual class in androidMain
- Bundle `Malaria_BNLeaky_Keras.tflite`
- Throwaway test harness on Android
- End-to-end inference test, results match iOS within float tolerance

### Phase 4 — iOS persistence (1.5 weeks)

- Author `docs/SCHEMA.md`
- SwiftData `@Model` classes mirroring schema
- `ModelContainerFactory` with `NSFileProtectionComplete`
- Repository wrappers
- Audit log writer
- Unit tests and verification of file protection
- Schema-drift test

### Phase 5 — Android persistence (1.5 weeks)

- Room `@Entity` classes mirroring `SCHEMA.md`
- `MalariaDatabase` with SQLCipher
- `SecureKeyStore` with StrongBox preference and software fallback
- DAO implementations
- Audit log writer
- Instrumented tests, verify encryption by inspecting DB file
- Schema-drift test

### Phase 6 — iOS identity, auth, and onboarding (2 weeks)

- `AuthGate` session management
- `LAContext` biometric prompts
- Auto-logout timer using shared `SessionTimer`
- `ClinicianProfile` flow
- Both onboarding phases (admin + microscopist)
- Re-onboarding flow
- All onboarding audit entries

### Phase 7 — Android identity, auth, and onboarding (2 weeks)

- Equivalent `AuthGate` for Android
- `BiometricPrompt` integration
- Auto-logout using same shared `SessionTimer`
- Onboarding flows mirroring iOS exactly
- Re-onboarding
- All onboarding audit entries

### Phase 8 — Camera and live screening (3 weeks, parallel)

iOS half (1.5 weeks):
- AVFoundation integration
- Frame capture → ImageInput
- Active screening UI in Home tab
- Inline prediction display

Android half (1.5 weeks, can run in parallel if maintainer can context-switch):
- CameraX integration
- Frame capture → ImageInput
- Active screening UI in Home tab using Compose
- Inline prediction display

### Phase 9 — Override flow (1.5 weeks, parallel)

- Live override (both platforms)
- Review override (both platforms)
- Override persistence
- Audit entries
- UI for displaying overridden predictions

### Phase 10 — History and AI Analysis (2.5 weeks, parallel)

- Recent predictions list
- Flagged for review filter
- Sessions list and detail
- Audit log viewer with filters
- AI Analysis detail view
- Mark-as-duplicate flow
- Session relabeling with PII warning

### Phase 11 — Settings and About (2 weeks, parallel)

- All Settings sections
- Edit-triggers-biometric pattern
- Model picker UI
- Hugging Face downloader with progress (URLSession on iOS, OkHttp+WorkManager on Android)
- Cache management
- About tab
- Crash log viewer

### Phase 12 — Localization

**Cancelled.** Repository ships English-only. Crowdin scaffolding remains in place at the repo root (`crowdin.yml`, empty `values-{sw,fr,pt}/` directories, iOS xcstrings `sw` / `fr` / `pt` locale entries) for a downstream deployer who chooses to revive translation; the maintainer does not pursue a translation round. The language picker and on-disk locale preference machinery (Section 15) remain wired so a fork can light up additional locales without re-plumbing.

### Phase 13 — Export (1.5 weeks)

- Export bundle generation on both platforms
- HMAC signature via shared module
- Share sheet integration (UIActivityViewController on iOS; ACTION_SEND on Android)
- Export audit entries
- Bundle compatibility test: bundle exported from iOS opens and inspects identically on Android

### Phase 14 — Crash logs (1 week)

- Signal-safe writer on iOS
- JVM + native crash handler on Android
- Crash log viewer in Settings on both platforms
- 30-day auto-expiry
- Sharing flow

### Phase 15 — Test hardening (2 weeks)

- Unit test coverage to target on shared module
- Coverage to target on both platform-specific layers
- Snapshot tests on iOS for every screen and locale
- Compose preview tests on Android for every screen and locale
- Integration tests for inference on both platforms
- Manual test plan execution on real iOS device + Pixel emulator/device

### Phase 16 — Documentation (1.5 weeks)

- Complete: KMP_App_Specification (this doc), SCHEMA, COMPLIANCE, ARCHITECTURE, Technical_Glossary_for_Beginners, MANUAL_TEST_PLAN
- README with screenshots from both platforms
- CONTRIBUTING.md with both platforms' build instructions

### Phase 17 — v0.1 launch preparation (1 week)

- Flip repo to public
- Tag v0.1.0
- Hugging Face model uploads complete (both .mlpackage and .tflite per remote model)
- TestFlight build available (iOS)
- Play Internal Testing track available (Android)
- Personal-channels announcement

### Phase 18 — Clinical advisor review and v1.0

**Cancelled.** v1.0 is reachable at maintainer discretion without an external clinical-advisor sign-off gate. The medical-device disclaimer in `NOTICE` carries the safety framing; deployers seeking clinical validation under their own jurisdiction add it as part of their conformance work (spec §18).

**Total to v0.1: ~30 weeks.** v1.0 timeline is whatever the maintainer subsequently commits to; the conservative criterion is "tagged when the maintainer judges the implementation stable and the spec deliverables complete." There is no external dependency between v0.1 and v1.0.

This is roughly 75% longer than an iOS-only build. The extra time reflects the genuine cost of feature parity across two platforms — not a doubling because the shared module work is done once.

---

## 23. Public launch checklist

### v0.1 launch (technical-quality)

Repository:
- [ ] Repo flipped from private to public
- [ ] README complete with screenshots from both iOS and Android and current-status banner
- [ ] LICENSE (Hippocratic 3.0) committed
- [ ] NOTICE (medical disclaimer) committed
- [ ] CONTRIBUTING.md with build instructions for both platforms
- [ ] CODE_OF_CONDUCT.md (Contributor Covenant)
- [ ] SECURITY.md with private disclosure email
- [ ] All `docs/` files complete
- [ ] `.github/ISSUE_TEMPLATE/` populated
- [ ] All five CI workflows passing on `main`

External:
- [ ] Hugging Face repo published with all 17 remote models in both .mlpackage and .tflite formats
- [ ] Each model has SHA-256 hash recorded per platform
- [ ] `model_registry.json` updated with Hugging Face URLs and per-platform hashes
- [ ] TestFlight build available for community testers (iOS)
- [ ] Play Internal Testing track available for community testers (Android)

Quality:
- [ ] All Phase 1–17 acceptance criteria met
- [ ] Test coverage at target on shared and both platforms
- [ ] No `// TODO` or `// FIXME` left in production code
- [ ] Manual test plan executed on at least one real iOS device and one real Android device

Communications:
- [ ] Personal LinkedIn post linking to repo
- [ ] Maintainer-affiliated channels (academic, professional) informed
- [ ] Explicit "research prototype — not a medical device, not for clinical use" framing everywhere (NOTICE language carries the safety framing)

### v1.0 launch

v1.0 is tagged when the maintainer judges the implementation stable and the spec deliverables complete. There is no external sign-off gate.

- [ ] All spec §22 phases that remain in scope are complete (Phases 12 and 18 are cancelled and excluded from this checklist)
- [ ] All v0.1 launch items above remain green
- [ ] Tag v1.0.0
- [ ] Release notes document the gate criteria the maintainer applied
- [ ] Optional: demo video produced and linked (preferably showing both platforms)

### What does NOT happen at v0.1 or v1.0

- No Hacker News post
- No Reddit submission
- No paid advertising
- No press outreach
- No App Store / Play Store production submission by the maintainer

Distribution is via GitHub clone and TestFlight/Play Internal invitation for v0.x; v1.0 may transition to App Store and Play Store if a deployer-affiliated developer account becomes available, but that's not a maintainer commitment.

---

## 24. Open questions

This section reflects the resolved status of all open items identified during spec design. Items are categorized by the maintainer's commitment level: planned for a specific version, deferred indefinitely (fork territory), or evidence-gated for v2.

### Blocking v1.0

**None.** Prior versions of this spec named clinical-advisor sign-off as the gate between v0.x and v1.0. That requirement has been removed from the project scope. v1.0 is now reachable at maintainer discretion when the implementation is judged stable and the spec deliverables in scope are complete. The medical-device disclaimer in `NOTICE` is the project's safety framing in lieu of an external advisor review.

A deployer seeking clinical validation under their own jurisdiction undertakes that work as part of their own conformance posture (§18); that work is deployer-fork territory, not an upstream maintainer commitment. See the "Deferred indefinitely — deployer-fork territory" subsection below.

### Planned for v1.1

**Import of export bundles.** v1.1 commitment, modest scope. The export format (Section 14) is already designed to be importable; v1.1 adds the import parser, signature validation, entity persistence, and audit trail. Cross-platform import (iOS bundle → Android, and vice versa) is an explicit goal, made possible by the byte-identical export format.

Imported predictions preserve their original `actorId` rather than being re-attributed to the importing clinician. The receiving device becomes effectively multi-actor for those imported rows while remaining single-clinician for new predictions. Imported actor IDs are read-only (Section 9). A new audit action `BUNDLE_IMPORTED` records the import event with metadata listing the imported predictions' actor IDs and the original device UUID.

Use case is rare device-migration scenarios. Honest framing: the feature exists for completeness, not because clinics are clamoring for it.

**Explicit sessions with pseudonymous patient refs.** v1.1 commitment as a proper schema migration. v1 schema is implicit-sessions-only with no reserved fields; v1.1 adds `sessionMode` and `patientRef` columns plus a new `Session` entity via SwiftData's `VersionedSchema` and Room's `Migration` infrastructure.

Real deployments may prefer explicit "Start session for patient X" workflow. The pseudonymous patient-ref design avoids PII risk by storing only an externally-linked UUID, never a name or identifier that maps to a person inside the app.

### Planned for v2

**Multi-clinician device support.** v2 commitment with design sketch documented now (rather than deferred to v2-design-time). The v1 single-clinician + override-time-initials choice is documented as v1-specific architecture, not "the final design."

v2 multi-clinician design sketch:

- *Identity:* multiple `ClinicianProfile` rows per device. A new "active profile" pointer tracks current authentication. Profile switching requires re-authentication.
- *Profile selection:* happens at the post-biometric step. Biometric confirms "an authorized user is here"; profile picker confirms "which authorized user." Biometric does not uniquely identify the profile.
- *Auth model:* session is bound to active profile. Profile switch creates a new session. Session-relock returns to profile picker (multi-profile) or biometric (single-profile).
- *Audit model:* all audit entries continue recording the active `actorId`. New action `PROFILE_SWITCHED` records profile changes within a session. No schema change to `AuditEntry` itself.
- *Permissions:* per-profile role assignment, set during enrollment. No self-elevation; only an admin profile can add new profiles.
- *Onboarding:* Phase 1 (admin provisioning) gains an opt-in "Enable multi-clinician mode" toggle. Default off (preserves v1 behavior). When on, "Enroll additional microscopists" flow accessed from Settings, admin-authenticated, each microscopist completes a Phase 2 claim flow creating a new profile rather than replacing the existing one.

v2 work begins from this sketch, not from scratch.

### Deferred indefinitely — not on maintainer roadmap (deployer-fork territory)

**Localization to non-English locales.** The repository ships English-only. The Crowdin scaffolding committed earlier in the project's history (`crowdin.yml` at the repo root, empty `androidApp/src/main/res/values-{sw,fr,pt}/` directories, iOS `Localizable.xcstrings` `sw` / `fr` / `pt` locale entries) remains in place as harmless future-proofing. The maintainer does not solicit translations and the Crowdin project is not provisioned. A deployer who wishes to revive translation under their own fork can do so without re-plumbing the locale machinery — the language picker, on-disk preference, and per-locale resource directories are already wired. See Section 15.

**Right-to-left language capability.** The v1 codebase is written RTL-ready (Section 11) as engineering discipline — semantic alignment, auto-mirrored icons, no absolute directional values. No RTL language ships in v1.x. A deployer-fork that translates to an RTL language enables the capability; because the layout discipline is already in force, the work is mostly verification rather than re-implementation.

**Regulatory conformance and clinical validation.** A deployer who wants to use this software in a regulated clinical setting takes on the conformance burden for their jurisdiction — clinical validation studies, DPIA / BAA where applicable, Notified Body conformity assessment (EU MDR), FDA 510(k) (US), ISO 13485 quality management, IEC 62304 medical device software lifecycle. The maintainer provides the application; the deployer provides the regulatory posture. `docs/COMPLIANCE.md` enumerates the deferred items in full.

**Cloud tier.** Same treatment as hard delete. Design reference preserved in `docs/CLOUD_TIER_REFERENCE.md` for forkers. The shared `Classifier` interface supports a parallel `CloudClassifier` without architectural changes (Section 5). Compliance burden (BAA, mTLS, region routing, key rotation, breach response) is too deployment-specific for the maintainer to generalize. A deployer adding cloud tier extends rather than modifies.

**Hard delete.** A clinic with a strong right-to-erasure obligation under their jurisdiction's law adds hard-delete to their fork. Hard delete is regulatory policy, not a feature — the maintainer can't decide for unknown deployers what conditions allow deletion. `docs/COMPLIANCE.md` documents the implementation pattern (where in the code to hook, what audit events to log, admin-authentication requirements) so a deployer can add it cleanly.

**App Store / Play Store production submission.** Generic release workflows produce signed builds with placeholder developer-account configuration. A deployer fills in their own team ID, signing certificates, and store credentials in their fork, then submits under their own developer account. `docs/STORE_SUBMISSION.md` documents the placeholder pattern, deployer responsibilities, and the requirement that store-listing data-handling disclosures reflect the deploying organization (not the upstream maintainer). Maintainer does not lead store submission.

**Schema drift code generator.** Dropped from the roadmap entirely. CI snapshot tests on both platforms (Section 20) are the official mitigation for schema drift between SwiftData and Room. The 4-entity schema is small enough that maintaining a code generator costs more than it saves. `docs/SCHEMA.md` notes the conditions under which to reconsider: substantial schema growth (say, 10+ entities) or frequent migrations.

### v2-conditional (evidence-gated)

**Compose Multiplatform reconsideration.** Decided against in v1. The v1 native-UI architecture stands. v2 reconsideration is evidence-gated, not pre-committed: if dual-UI maintenance burden proves unsustainable based on real v1.x deployment evidence, specific screens (AI Analysis view, Audit Log viewer) may migrate from native-per-platform to Compose Multiplatform.

The shared-module/native-UI boundary itself is fixed — the only question would be whether some screens within the native-UI layer migrate, not whether the whole UI layer changes paradigm. Compose Multiplatform's iOS support continues to mature; the 2026 calculus may not be the 2027+ calculus.

---

The shape of this section has evolved during spec design from "list of things we might do someday" to honest categorization: things the maintainer will do, things they won't, the conditions under which to reconsider. Each item names its commitment level explicitly.

---

## 25. References

### Project artifacts
- ML notebook (Part 7): Core ML and TFLite export pipelines, model_registry.json schema, Vision/Core ML inference Swift snippets
- Presentation deck Appendix C: UI mockups for the original six screens
- `docs/Technical_Glossary_for_Beginners.md`: glossary for non-ML readers

### Technical — cross-platform
- Kotlin Multiplatform Mobile — https://kotlinlang.org/lp/multiplatform/
- Hugging Face Hub — https://huggingface.co/docs/hub
- Hippocratic License 3.0 — https://firstdonoharm.dev/

### Technical — iOS
- SwiftData — https://developer.apple.com/documentation/swiftdata
- Swift Package Manager — https://www.swift.org/documentation/package-manager/
- Core ML — https://developer.apple.com/documentation/coreml
- Vision framework — https://developer.apple.com/documentation/vision
- AVFoundation — https://developer.apple.com/documentation/avfoundation
- LocalAuthentication — https://developer.apple.com/documentation/localauthentication
- iOS Data Protection — https://support.apple.com/guide/security/data-protection-classes-secb010e978a/web
- Xcode String Catalogs — https://developer.apple.com/documentation/xcode/localizing-and-varying-text-with-a-string-catalog

### Technical — Android
- Jetpack Compose — https://developer.android.com/jetpack/compose
- Material 3 Expressive — https://m3.material.io/
- Room — https://developer.android.com/jetpack/androidx/releases/room
- SQLCipher for Android — https://www.zetetic.net/sqlcipher/sqlcipher-for-android/
- Android Keystore + StrongBox — https://developer.android.com/privacy-and-security/keystore
- BiometricPrompt — https://developer.android.com/reference/androidx/biometric/BiometricPrompt
- CameraX — https://developer.android.com/training/camerax
- LiteRT (TFLite) — https://ai.google.dev/edge/litert
- WorkManager — https://developer.android.com/topic/libraries/architecture/workmanager

### Compliance (informational, for deployers)
- GDPR (Regulation 2016/679) — https://gdpr-info.eu/
- HIPAA Security Rule (45 CFR Part 164 Subpart C) — https://www.hhs.gov/hipaa/
- EU MDR 2017/745 — https://eur-lex.europa.eu/eli/reg/2017/745/oj
- FDA SaMD guidance — https://www.fda.gov/medical-devices/software-medical-device-samd
- Kenya Data Protection Act 2019 — https://www.odpc.go.ke/

### Standards bodies (for deployers)
- ISO 13485 (medical device QMS) — https://www.iso.org/standard/59752.html
- IEC 62304 (medical device software lifecycle) — https://www.iso.org/standard/38421.html
- ISO 27799 (health informatics security) — https://www.iso.org/standard/62777.html

---

**End of specification.**

This document supersedes any prior version of `KMP_App_Specification.md`. Changes to this document follow the same review process as code changes: a PR with at least one maintainer review, signed commits required for releases, semantic versioning of the spec itself (the spec version equals the targeted app version).
