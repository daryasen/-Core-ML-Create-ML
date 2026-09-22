import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject private var classifier = ImageClassifier()
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var didLoadSample = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    header
                    imageCard
                    actionButton
                    resultCard
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("ML Vision")
            .onChange(of: selectedItem) { _, item in
                Task { await load(item) }
            }
            .task {
                guard !didLoadSample,
                      let path = Bundle.main.path(forResource: "SampleDog", ofType: "jpg"),
                      let sample = UIImage(contentsOfFile: path) else { return }
                didLoadSample = true
                selectedImage = sample
                await classifier.classify(sample)
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Core ML + Vision", systemImage: "brain.head.profile")
                .font(.title2.bold())
                .foregroundStyle(.blue)
            Text("MobileNetV2 распознаёт изображение локально — фото не покидает устройство.")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var imageCard: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 24)
                .fill(.background)
                .shadow(color: .black.opacity(0.06), radius: 12, y: 4)

            if let selectedImage {
                Image(uiImage: selectedImage)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .padding(8)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 48))
                        .foregroundStyle(.blue)
                    Text("Выберите фото для анализа")
                        .font(.headline)
                }
            }
        }
        .frame(height: 220)
    }

    private var actionButton: some View {
        PhotosPicker(selection: $selectedItem, matching: .images) {
            Label(selectedImage == nil ? "Выбрать изображение" : "Выбрать другое", systemImage: "photo.badge.plus")
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .background(.blue, in: RoundedRectangle(cornerRadius: 16))
                .foregroundStyle(.white)
        }
    }

    @ViewBuilder
    private var resultCard: some View {
        if classifier.isRunning {
            ProgressView("Анализ изображения…")
                .frame(maxWidth: .infinity)
                .padding()
        } else if let error = classifier.errorMessage {
            Label(error, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 16))
        } else if !classifier.predictions.isEmpty {
            VStack(alignment: .leading, spacing: 14) {
                Text("Результат")
                    .font(.headline)
                ForEach(Array(classifier.predictions.enumerated()), id: \.offset) { index, prediction in
                    HStack {
                        Text("\(index + 1)")
                            .font(.caption.bold())
                            .frame(width: 28, height: 28)
                            .background(.blue.opacity(0.12), in: Circle())
                        Text(prediction.label.replacingOccurrences(of: "_", with: " "))
                            .lineLimit(2)
                        Spacer()
                        Text(prediction.confidence, format: .percent.precision(.fractionLength(1)))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding()
            .background(.background, in: RoundedRectangle(cornerRadius: 20))
        }
    }

    @MainActor
    private func load(_ item: PhotosPickerItem?) async {
        guard let data = try? await item?.loadTransferable(type: Data.self),
              let image = UIImage(data: data) else { return }
        selectedImage = image
        await classifier.classify(image)
    }
}
