import XCTest

/// Screenshot capture tests. Each test navigates to a screen and calls
/// `snapshot(_:)` — the XCTAttachment is saved to the test result bundle
/// and can be exported from Xcode's Report Navigator.
///
/// Run on a connected device or simulator that matches the target
/// screen size. The app launches with `--skip-onboarding` (wired in
/// `MalariaDetectorUITestCase`) so no database seeding is needed.
final class ScreenshotTests: MalariaDetectorUITestCase {

    func testScreenshot_Home() {
        snapshot("home")
    }

    func testScreenshot_History() {
        app.tabBars.buttons.element(boundBy: 1).tap()
        snapshot("history")
    }

    func testScreenshot_Settings() {
        app.tabBars.buttons.element(boundBy: 2).tap()
        snapshot("settings")
    }
}
