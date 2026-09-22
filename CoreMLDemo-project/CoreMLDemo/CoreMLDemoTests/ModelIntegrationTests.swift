import CoreML
import XCTest
@testable import CoreMLDemo
import UIKit

@MainActor
final class ModelIntegrationTests: XCTestCase {
    func testConvertedModelLoads() throws {
        let model = try MobileNetV2(configuration: MLModelConfiguration())
        XCTAssertEqual(model.model.modelDescription.inputDescriptionsByName["image"]?.type, .image)
    }

    func testBundledPhotoIsClassified() async throws {
        let path = try XCTUnwrap(Bundle.main.path(forResource: "SampleDog", ofType: "jpg"))
        let image = try XCTUnwrap(UIImage(contentsOfFile: path))
        let classifier = ImageClassifier()

        await classifier.classify(image)

        XCTAssertNil(classifier.errorMessage)
        XCTAssertEqual(classifier.predictions.count, 3)
        XCTAssertTrue(classifier.predictions.allSatisfy { (0...1).contains($0.confidence) })
    }
}
