from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


CLASS_MAP = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download and split the Animals-10 dataset into folders.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="Rapidata/Animals-10")
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument("--valid-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    dataset = load_dataset(args.dataset_name, split="train")

    grouped_indices = defaultdict(list)
    label_feature = dataset.features["label"]
    for index, sample in enumerate(dataset):
        raw_label = label_feature.int2str(sample["label"])
        label = CLASS_MAP.get(raw_label.lower(), raw_label.lower())
        grouped_indices[label].append(index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_sizes = {
        "train": args.train_per_class,
        "valid": args.valid_per_class,
        "test": args.test_per_class,
    }

    summary = {}
    for label, indices in grouped_indices.items():
        rng.shuffle(indices)
        required = sum(split_sizes.values())
        selected = indices[:required]
        offset = 0
        summary[label] = {}
        for split_name, split_size in split_sizes.items():
            split_dir = args.output_dir / split_name / label
            split_dir.mkdir(parents=True, exist_ok=True)
            split_indices = selected[offset : offset + split_size]
            for image_number, dataset_index in enumerate(split_indices):
                image = dataset[dataset_index]["image"].convert("RGB")
                image.save(split_dir / f"{label}_{image_number:04d}.jpg", quality=95)
            offset += split_size
            summary[label][split_name] = len(split_indices)

    print(summary)


if __name__ == "__main__":
    main()
