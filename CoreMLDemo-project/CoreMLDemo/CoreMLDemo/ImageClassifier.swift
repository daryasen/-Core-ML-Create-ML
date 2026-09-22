import CoreML
import UIKit
import Vision

struct Prediction {
    let label: String
    let confidence: Double
}

@MainActor
final class ImageClassifier: ObservableObject {
    @Published var predictions: [Prediction] = []
    @Published var isRunning = false
    @Published var errorMessage: String?

    func classify(_ image: UIImage) async {
        guard let cgImage = image.cgImage else {
            errorMessage = "Не удалось прочитать изображение."
            return
        }

        isRunning = true
        errorMessage = nil
        predictions = []

        do {
            let result = try await Self.performRequest(for: cgImage)
            predictions = result.prefix(3).map {
                Prediction(label: $0.identifier, confidence: Double($0.confidence))
            }
        } catch {
            errorMessage = "Ошибка классификации: \(error.localizedDescription)"
        }
        isRunning = false
    }

    private nonisolated static func performRequest(for image: CGImage) async throws -> [VNClassificationObservation] {
        try await withCheckedThrowingContinuation { continuation in
            do {
                let configuration = MLModelConfiguration()
                configuration.computeUnits = .all
                let model = try MobileNetV2(configuration: configuration).model
                let visionModel = try VNCoreMLModel(for: model)
                let request = VNCoreMLRequest(model: visionModel) { request, error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else if let observations = request.results as? [VNClassificationObservation] {
                        continuation.resume(returning: observations)
                    } else {
                        continuation.resume(throwing: ClassifierError.noResults)
                    }
                }
                request.imageCropAndScaleOption = .centerCrop
                try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

private enum ClassifierError: LocalizedError {
    case noResults
    var errorDescription: String? { "Модель не вернула результат." }
}
