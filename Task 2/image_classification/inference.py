from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def load_model(model_dir: str | Path, device):
    model_dir = Path(model_dir)
    class_names = json.loads((model_dir / "class_names.json").read_text(encoding="utf-8"))
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


def predict_image(image_path: str | Path, model_dir: str | Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(model_dir, device)
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).item()
    return class_names[prediction]


def parse_args():
    parser = argparse.ArgumentParser(description="Run image classifier inference.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    predicted_class = predict_image(args.image_path, args.model_dir)
    print({"predicted_class": predicted_class})


if __name__ == "__main__":
    main()
