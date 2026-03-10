from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ANIMALS = {
    "cat": ["cat", "cats", "kitten", "kittens", "feline"],
    "dog": ["dog", "dogs", "puppy", "puppies", "canine"],
    "cow": ["cow", "cows", "calf", "cattle"],
    "horse": ["horse", "horses", "pony", "mare"],
    "sheep": ["sheep", "lamb", "lambs", "ewes"],
    "elephant": ["elephant", "elephants"],
    "butterfly": ["butterfly", "butterflies"],
    "chicken": ["chicken", "chickens", "hen", "rooster"],
    "spider": ["spider", "spiders"],
    "squirrel": ["squirrel", "squirrels"],
}

TEMPLATES = [
    "There is a {animal} in the picture .",
    "I think the photo shows a {animal} .",
    "The image contains a {animal} near the center .",
    "It looks like a {animal} is visible here .",
    "My guess is that this picture has a {animal} .",
    "Can you confirm that the animal is a {animal} ?",
    "I can clearly see a {animal} in this image .",
    "The user says the uploaded image contains a {animal} .",
]

MULTI_TEMPLATES = [
    "There might be a {animal_a} and a {animal_b} in the same picture .",
    "I can spot a {animal_a} while another {animal_b} is in the background .",
]

NEGATIVE_TEMPLATES = [
    "I cannot recognize the animal in the image .",
    "The picture is too blurry to identify anything .",
    "There is no visible animal in this frame .",
]


def to_bio(tokens, entity_tokens):
    labels = ["O"] * len(tokens)
    for start in range(len(tokens) - len(entity_tokens) + 1):
        if tokens[start : start + len(entity_tokens)] == entity_tokens:
            labels[start] = "B-ANIMAL"
            for index in range(1, len(entity_tokens)):
                labels[start + index] = "I-ANIMAL"
            return labels
    raise ValueError(f"Could not align entity tokens {entity_tokens} in sentence {tokens}")


def build_positive_examples(rng: random.Random):
    samples = []
    for canonical, aliases in ANIMALS.items():
        for alias in aliases:
            for template in TEMPLATES:
                tokens = template.format(animal=alias).split()
                entity_tokens = alias.split()
                samples.append({"tokens": tokens, "ner_tags": to_bio(tokens, entity_tokens), "label": canonical})
    animal_names = list(ANIMALS)
    for animal_a in animal_names:
        for animal_b in animal_names:
            if animal_a == animal_b:
                continue
            template = rng.choice(MULTI_TEMPLATES)
            alias_a = rng.choice(ANIMALS[animal_a])
            alias_b = rng.choice(ANIMALS[animal_b])
            sentence = template.format(animal_a=alias_a, animal_b=alias_b)
            tokens = sentence.split()
            labels = ["O"] * len(tokens)
            for alias in (alias_a, alias_b):
                entity_tokens = alias.split()
                entity_labels = to_bio(tokens, entity_tokens)
                labels = [
                    current if current != "O" else candidate
                    for current, candidate in zip(labels, entity_labels)
                ]
            samples.append({"tokens": tokens, "ner_tags": labels, "label": f"{animal_a},{animal_b}"})
    return samples


def build_negative_examples():
    return [{"tokens": sentence.split(), "ner_tags": ["O"] * len(sentence.split()), "label": "none"} for sentence in NEGATIVE_TEMPLATES]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"tokens": row["tokens"], "ner_tags": row["ner_tags"]}, ensure_ascii=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic NER data for animal extraction.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows = build_positive_examples(rng) + build_negative_examples()
    rng.shuffle(rows)

    total = len(rows)
    train_end = int(total * args.train_ratio)
    valid_end = train_end + int(total * args.valid_ratio)

    write_jsonl(args.output_dir / "train.jsonl", rows[:train_end])
    write_jsonl(args.output_dir / "valid.jsonl", rows[train_end:valid_end])
    write_jsonl(args.output_dir / "test.jsonl", rows[valid_end:])

    print(
        {
            "train": train_end,
            "valid": valid_end - train_end,
            "test": total - valid_end,
            "total": total,
        }
    )


if __name__ == "__main__":
    main()
