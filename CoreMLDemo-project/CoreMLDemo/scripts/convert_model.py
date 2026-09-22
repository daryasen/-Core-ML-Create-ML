"""Download MobileNetV2 from PyTorch Hub and convert it to Core ML."""
from pathlib import Path
import urllib.request

import coremltools as ct
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "CoreMLDemo" / "MobileNetV2.mlpackage"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def main() -> None:
    # The model is loaded directly from the official PyTorch Vision Hub repo.
    model = torch.hub.load("pytorch/vision:v0.25.0", "mobilenet_v2", weights="DEFAULT")
    model.eval()

    class NormalizedModel(nn.Module):
        def __init__(self, network: nn.Module) -> None:
            super().__init__()
            self.network = network
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            logits = self.network((image - self.mean) / self.std)
            return torch.softmax(logits, dim=1)

    model = NormalizedModel(model).eval()
    example = torch.rand(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)

    labels_path = ROOT / "scripts" / "imagenet_classes.txt"
    if not labels_path.exists():
        urllib.request.urlretrieve(LABELS_URL, labels_path)
    labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]

    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(
            name="image",
            shape=example.shape,
            scale=1 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        )],
        classifier_config=ct.ClassifierConfig(labels),
        minimum_deployment_target=ct.target.iOS17,
    )
    converted.author = "PyTorch Vision / converted with coremltools"
    converted.short_description = "MobileNetV2 ImageNet image classifier"
    converted.save(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
