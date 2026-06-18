import XCTest

/// Base class for all UI tests. Launches the app with `--skip-onboarding`
/// so the onboarding wizard is bypassed and `RootView` is shown immediately
/// on a clean (empty) store.
class MalariaDetectorUITestCase: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--skip-onboarding"]
        app.launch()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    /// Captures a screenshot and attaches it to the test result with the
    /// given name. Use this for both manual review and automated diffing.
    func snapshot(_ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
