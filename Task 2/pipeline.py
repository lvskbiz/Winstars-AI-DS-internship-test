from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from image_classification.inference import predict_image
from ner.inference import extract_animals


def run_pipeline(text: str, image_path: str | Path, ner_model_dir: str | Path, image_model_dir: str | Path) -> bool:
    extracted_animals = extract_animals(text, ner_model_dir)
    predicted_animal = predict_image(image_path, image_model_dir)
    return predicted_animal in extracted_animals


def parse_args():
    parser = argparse.ArgumentParser(description="Compare animal mention in text against image classification output.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--ner-model-dir", type=Path, required=True)
    parser.add_argument("--image-model-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_pipeline(
        text=args.text,
        image_path=args.image_path,
        ner_model_dir=args.ner_model_dir,
        image_model_dir=args.image_model_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
