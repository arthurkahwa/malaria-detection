// Intentionally empty.
//
// Per spec §4 each service is injected via SwiftUI environment from the
// composition root. The spec example used the legacy `EnvironmentKey`
// pattern with a `static let defaultValue = { fatalError(...) }()` to
// fail loudly on missing wiring. That pattern is unsafe under iOS 17+
// SwiftUI: `WritableKeyPath._projectMutableAddress` reads `defaultValue`
// during SET, before the chained value is assigned, so the fatalError
// fires every time the chain initialises.
//
// The fix is to use the modern type-based form:
//
//   composition root: `.environment(authGate)`     // pass the instance
//   consumer view:    `@Environment(AuthGate.self) private var authGate`
//
// SwiftUI keys the value by the type's identity; no `EnvironmentKey` is
// required at all, which means no `defaultValue` to trip the assertion.
// Each `@Observable @MainActor` service in `Services/` is injected this
// way from `MalariaDetectorApp`.
//
// The raw `Shared.Classifier` (Kotlin/Native generated, not `@Observable`)
// is wrapped by `ClassifierBridge` for injection.
