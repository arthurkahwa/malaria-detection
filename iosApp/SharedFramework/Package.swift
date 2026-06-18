// swift-tools-version:6.0
import PackageDescription

// `.v26` was added in Swift PM 6.2 (Xcode 26). Use the string form so
// the manifest parses on older toolchains too — Xcode 26 will accept it.
let package = Package(
    name: "Shared",
    platforms: [.iOS("26.0")],
    products: [.library(name: "Shared", targets: ["Shared"])],
    targets: [.binaryTarget(name: "Shared", path: "Shared.xcframework")]
)
