# Contributing to Malaria Detector

Thank you for considering a contribution. This is a Kotlin Multiplatform
project with native UIs on both iOS (SwiftUI + SwiftData) and Android
(Compose + Room). The full design lives in
[`KMP_App_Specification.md`](./KMP_App_Specification.md).

## Quick start

1. Clone the repository.
2. Generate the Gradle wrapper if you have not already:
   ```sh
   gradle wrapper
   ```
   The wrapper jar/properties are intentionally not checked in for the initial
   scaffold — run `gradle wrapper` once after cloning.
3. Build the shared XCFramework that the iOS app consumes:
   ```sh
   ./gradlew :shared:assembleSharedXCFramework
   ```
4. **iOS:** generate the Xcode project from `iosApp/project.yml`:
   ```sh
   xcodegen generate --spec iosApp/project.yml --project iosApp
   ```
   Open the resulting `iosApp/MalariaDetector.xcodeproj` in Xcode.
5. **Android:** open `androidApp/` in Android Studio (Iguana or newer). The
   shared module is consumed via `implementation(project(":shared"))`.

## Translations

The project ships English-only and the maintainer does not currently
solicit translation contributions. Crowdin scaffolding
([`crowdin.yml`](./crowdin.yml), the `values-{sw,fr,pt}/` locale
directories under `androidApp/src/main/res/`, and the `sw` / `fr` /
`pt` entries inside `iosApp/Localization/Localizable.xcstrings`)
remains in the repo for a downstream deployer who wishes to revive
this effort under their own fork. See spec §15 for the rationale and
§24 for the deployer-fork framing.

## Code standards

- **Kotlin:** `ktlint` is the formatter and linter. CI runs `ktlintCheck` on
  every PR.
- **Swift:** `SwiftLint` runs in CI as a Phase 17 deliverable. Until it is
  wired in, follow the Swift API Design Guidelines and the patterns used in
  existing `iosApp/` code.

## Architecture decisions

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for the layering rules
(shared Kotlin / native persistence / native UI), the no-ViewModels
environment-injection pattern, and the concurrency model.

## Reporting issues

Please use the templates in [`.github/ISSUE_TEMPLATE/`](./.github/ISSUE_TEMPLATE/).
Security issues go to the address in [`SECURITY.md`](./SECURITY.md), not the
public tracker.
