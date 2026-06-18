# SCHEMA — Canonical Persistence Schema

> **Status: iOS implementation in place (Phase 4); Android implementation pending (Phase 5)**

This document is the single source of truth for the malaria-detector
persistence schema. The schema is defined here once and implemented twice: as
SwiftData `@Model` classes on iOS (`iosApp/Models/`) and as Room `@Entity`
classes on Android (`androidApp/.../data/entities/`). Schema drift between
implementations is the responsibility of the schema-drift CI snapshot tests
(see "Schema-drift CI mitigation" below) — they are the official mitigation,
not a code generator.

Source: `KMP_App_Specification.md` §8.

## Canonical schema overview

Four persisted entities, identical structure on both platforms:

- `Prediction` — one row per inference event
- `AuditEntry` — append-only event log
- `ClinicianProfile` — the device's single clinician (v1) — schema TBD by Phase 4
- `ConsentRecord` — onboarding consent acknowledgements — schema TBD by Phase 4

All four entities are persisted under at-rest encryption:
`NSFileProtectionComplete` on iOS, SQLCipher AES-256 on Android. Images are
never persisted; there is no image field on `Prediction`.

## Prediction

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

Note: there is no `image` field, no `imageBlob`, no path to a stored image.
Images do not persist on either platform.

iOS implementation: `iosApp/Models/Prediction.swift`; Android implementation: `<TBD Phase 5>`

## AuditEntry

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

The audit log is **append-only** by application convention (no UPDATE or
DELETE SQL ever issued against it by app code) on both platforms. The
repositories enforce this by not exposing mutation or deletion methods on
the `AuditEntry` type — only `write(...)`.

### Canonical audit action vocabulary

Stored as canonical English strings regardless of UI locale (spec §8). The
iOS `AuditAction` enum at `iosApp/Models/AuditAction.swift` is the source of
truth for the canonical string mapping; Android mirrors it in Phase 5.
Adding a value is a versioning event; removing a value is breaking and not
done.

- **Lifecycle:** `admin_provisioning_started`, `clinic_configured`,
  `inference_policy_set`, `admin_biometric_enrolled`,
  `admin_provisioning_completed`, `microscopist_claim_started`,
  `microscopist_biometric_enrolled`, `microscopist_claim_completed`,
  `device_reprovisioned`, `role_transferred`, `profile_updated`
- **Authentication:** `auth_success`, `auth_failure`, `session_unlocked`,
  `session_relocked_background`, `session_relocked_manual`,
  `session_relocked_timeout`
- **Models:** `model_download_initiated`, `model_download_completed`,
  `model_download_failed`, `model_download_hash_mismatch`,
  `model_cache_cleared`, `active_model_changed`
- **Inference:** `prediction_created`, `prediction_viewed`,
  `override_recorded`, `prediction_linked_as_duplicate`,
  `session_relabeled`
- **Data management:** `export_initiated`, `export_completed`,
  `export_failed`, `crash_log_shared`
- **Configuration:** `threshold_changed`, `default_model_changed`,
  `auto_logout_changed`, `language_changed`

iOS implementation: `iosApp/Models/AuditEntry.swift`; Android implementation: `<TBD Phase 5>`

## ClinicianProfile

A single row per device installation (v1 is single-clinician). The minimal
shape used in domain logic is described in spec §9.

| Field | Type | Notes |
|-------|------|-------|
| id | String (UUID) | Primary key, equal to `actorId` |
| actorId | String (UUID) | Pseudonymous identifier generated at onboarding; never linked to PII inside the app |
| role | String | Canonical English: `admin` / `microscopist` / `observer` |
| initials | String? | Optional 2-character free text |
| enrolledAt | Date / Instant | When the profile was provisioned |
| biometricEnrolled | Bool | True after Phase 1 or Phase 2 biometric registration |

iOS implementation: `iosApp/Models/ClinicianProfile.swift`; Android implementation: `<TBD Phase 5>`

## ConsentRecord

One row per onboarding acknowledgement (Hippocratic License acknowledgement,
medical-device disclaimer acknowledgement, lawful-basis selection,
jurisdiction selection). Each ConsentRecord pairs with an `AuditEntry` for
the corresponding event.

| Field | Type | Notes |
|-------|------|-------|
| id | String (UUID) | Primary key |
| timestamp | Date / Instant | When acknowledged, UTC |
| actorId | String (UUID) | Who acknowledged (FK to `ClinicianProfile.actorId`) |
| consentType | String | Canonical English: `hippocratic_license` / `medical_disclaimer` / `lawful_basis` / `jurisdiction` |
| documentVersion | String | Version or content hash of the displayed text — locks the acknowledgement to specific wording |
| value | String | For `lawful_basis` and `jurisdiction`: the selected enum canonical string. For acknowledgement types: `"accepted"` |
| appVersion | String | At time of acknowledgement |

iOS implementation: `iosApp/Models/ConsentRecord.swift`; Android implementation: `<TBD Phase 5>`

## Schema-drift CI mitigation

A CI test on both platforms compares the actual entity definitions against
this document:

- **iOS:** SwiftData `@Model` reflection dumps the inferred schema; compared
  against a serialized form of this file.
- **Android:** Room's `@Database(exportSchema = true)` produces a JSON schema
  export; compared against this file.

Any drift fails the build on the affected platform. This is the official
mitigation; a schema-first code generator was considered and rejected
(spec §20). Reconsider only on substantial schema growth (10+ entities) or
frequent migrations.
