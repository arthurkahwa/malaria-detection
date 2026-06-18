import XCTest
@preconcurrency import Shared
@testable import MalariaDetector

/// End-to-end smoke test for the Core ML pipeline (spec §22 Phase 2):
/// builds a synthetic 128×128 RGB image, runs it through the bundled
/// `Malaria_BNLeaky_Keras` model, and asserts the output schema matches
/// what `PredictionStore` expects to map.
///
/// Probability values are not asserted because synthetic gray noise is
/// not a meaningful blood-cell image; we only verify that the model
/// loads, accepts the pixel buffer, runs inference, and returns
/// well-formed Parasitized/Uninfected probabilities that sum to ~1.0.
@MainActor
final class CoreMLClassifierTests: XCTestCase {

    func testBNLeakyKerasClassifier_runsSyntheticImage_endToEnd() async throws {
        let classifier = Classifier(modelId: "BNLeaky_Keras")

        // 128×128 RGB filled with mid-gray (0x80).
        let totalBytes: Int32 = 128 * 128 * 3
        let bytes = KotlinByteArray(size: totalBytes)
        for i in 0..<totalBytes {
            bytes.set(index: i, value: Int8(bitPattern: 0x80))
        }
        let image = ImageInput(rgbBytes: bytes, width: 128, height: 128)

        let prediction = try await classifier.classify(image: image)

        XCTAssertEqual(prediction.modelId, "BNLeaky_Keras")
        XCTAssertFalse(prediction.imageHash.isEmpty)
        XCTAssertGreaterThanOrEqual(prediction.inferenceMs, 0)

        let pSum = prediction.parasitizedProb + prediction.uninfectedProb
        XCTAssertEqual(Double(pSum), 1.0, accuracy: 0.01,
                       "Softmax outputs must sum to 1 (got \(pSum))")
        XCTAssertGreaterThanOrEqual(prediction.parasitizedProb, 0)
        XCTAssertLessThanOrEqual(prediction.parasitizedProb, 1)
    }
}
