import XCTest
@preconcurrency import Shared

/// Smoke test: confirm the shared XCFramework is reachable from
/// Swift and that the deterministic clinical-safety thresholds
/// behave as expected. Phase 1 + Phase 2 deliverable.
final class MalariaDetectorTests: XCTestCase {

    func testThresholdLogic_isCallable() {
        // `Threshold` in commonMain is a Kotlin `object`; the generated
        // Swift binding exposes it as `Threshold.shared`. Kotlin default
        // arguments don't survive the Swift bridge, so the `threshold`
        // parameter is required at the call site.
        let t = Threshold.shared
        XCTAssertEqual(t.label(parasitizedProb: 0.9, threshold: t.DEFAULT), "Parasitized")
        XCTAssertEqual(t.label(parasitizedProb: 0.1, threshold: t.DEFAULT), "Uninfected")
    }
}
