# COMPLIANCE — Posture, Implementation, Deferrals

> **Status: scaffold**

Source: `KMP_App_Specification.md` §18.

## Framing

This is a research-prototype open-source application licensed under
Hippocratic 3.0 with an explicit medical-device disclaimer in `NOTICE`. The
maintainer does **not** pursue regulatory clearance. A deployer who wants to
use this software in a clinical setting takes on the conformance burden for
their jurisdiction.

The application is built so the deployer's compliance work is **additive**,
not corrective. Encryption at rest, audit logging, biometric gating, and
no-image-persistence are implemented from v1. The architecture supports the
compliance features a deployer needs to add (chain hashing, auto-retention,
anomalous-access detection, mTLS for cloud tier) without restructuring the
codebase.

## What v1 implements (per platform)

Both platforms:

- At-rest encryption — `NSFileProtectionComplete` (iOS) and SQLCipher with
  Keystore-managed key (Android).
- Hardware-backed encryption keys — Secure Enclave (iOS), Android Keystore
  with StrongBox where the device supports it (software-backed Keystore
  otherwise).
- Biometric/passcode gate on app launch (`LAContext` on iOS,
  `BiometricPrompt` with `BIOMETRIC_STRONG | DEVICE_CREDENTIAL` on Android).
- Auto-logout after configurable inactivity (5/15/30 min; default 30,
  configured during admin onboarding).
- Complete audit log with the structured action vocabulary in §8.
- No image persistence.
- Pseudonymous clinician identity (UUID, never PII in-app).
- Override flow with attribution (single-clinician + free-text initials for
  multi-actor scenarios).
- Decision-support framing throughout — every prediction is overridable; the
  UI never presents a verdict as final.
- Lawful-basis capture at onboarding.
- Per-jurisdiction retention policy **displayed** in Settings.
- Onboarding consent acknowledgements with timestamped audit records.
- Signed export bundles.
- Privacy-preserving crash logs (no PHI, no third-party analytics).

### Android encryption-at-rest detail

Android v1 uses **SQLCipher** for the Room database, with the database
passphrase managed by `SecureKeyStore` (see
`androidApp/src/main/kotlin/com/malaria/android/data/SecureKeyStore.kt`). The
pattern is:

- On first launch, generate a random 32-byte SQLCipher passphrase.
- Generate an AES-256-GCM key in the **Android Keystore**, requesting
  StrongBox where the device supports it and falling back to a
  software-backed Keystore key otherwise.
- Encrypt the passphrase with that Keystore key and persist
  `iv || ciphertext` as a base64 string in SharedPreferences.
- On every subsequent launch, decrypt the passphrase via the Keystore key
  and hand it to SQLCipher.

The Keystore key never leaves the secure boundary; only the encrypted
passphrase blob is at rest in app-private storage. The SQLCipher passphrase
itself is held in memory only for the duration of the database open call.

## What v1 does not implement (deployer responsibilities)

These are listed verbatim from §18; the deployer assumes them as part of
their own conformance work:

- DPIA (GDPR Art. 35)
- BAA with any third-party service
- Cryptographic chain hashing of audit log with daily integrity verification
- Auto-enforcement of retention policy (delete records after N years)
- Anomalous-access detection
- Penetration testing
- App Store / Play Store medical-app submission
- Notified Body conformity assessment (EU MDR)
- FDA 510(k) submission (US)
- ISO 13485 quality management system
- IEC 62304 medical device software lifecycle compliance
- Privacy policy and terms of service review by qualified counsel per
  jurisdiction
- Clinical validation studies

## Platform-specific compliance surface

**iOS:**
- App Sandbox (default in iOS)
- App Transport Security: TLS 1.3 mandatory for cloud connections; ATS
  exception entries forbidden except for `huggingface.co`
- Privacy Manifest (`PrivacyInfo.xcprivacy`) declaring health data type, no
  tracking
- Privacy Nutrition Label declaring Health & Fitness, no tracking

**Android:**
- Scoped storage (default in API 36)
- Network Security Configuration: TLS pinning for cloud connections
  (informational in v1 since no cloud tier); cleartext blocked
- Data Safety section in Play Store listing declaring Health/Fitness,
  encryption in transit and at rest, retention policy
- `allowBackup="false"` with explicit data extraction rules preventing
  automatic cloud/device backup

## Hard-delete pattern (deployer-fork guide)

Hard delete is **not** implemented in v1 — it is regulatory policy, not a
feature, and the maintainer cannot decide for unknown deployers what
conditions allow deletion. A deployer with a strong right-to-erasure
obligation adds it to their fork. The architecture supports a clean
addition:

- Hook the delete at the platform repository layer (`PredictionRepository`
  on iOS, the `PredictionDao` wrapper on Android).
- Require an admin-authenticated biometric prompt before the delete is
  executed.
- Write an `OVERRIDE_RECORDED` / new `HARD_DELETE_EXECUTED` audit entry
  (deployers extend the action vocabulary; see §8).
- Treat the hard-delete code path as deployer-owned: gate it behind a
  build-time flag so the upstream main branch never ships it by default.

Full hook points and the audit-action extension pattern are documented here
in detail by Phase 17 authors.
