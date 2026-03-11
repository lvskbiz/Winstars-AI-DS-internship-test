from __future__ import annotations

import argparse
from pathlib import Path

from transformers import pipeline


CANONICAL_ANIMALS = {
    "cat": "cat",
    "cats": "cat",
    "dog": "dog",
    "dogs": "dog",
    "cow": "cow",
    "cows": "cow",
    "horse": "horse",
    "horses": "horse",
    "sheep": "sheep",
    "lion": "lion",
    "lions": "lion",
    "tiger": "tiger",
    "tigers": "tiger",
    "bear": "bear",
    "bears": "bear",
    "fox": "fox",
    "foxes": "fox",
    "wolf": "wolf",
    "wolves": "wolf",
}


def normalize_animal(token: str) -> str:
    normalized = token.lower().strip()
    return CANONICAL_ANIMALS.get(normalized, normalized)


def extract_animals(text: str, model_dir: str | Path):
    ner = pipeline(
        task="token-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        aggregation_strategy="simple",
    )
    animals = {
        normalize_animal(entity["word"])
        for entity in ner(text)
        if entity.get("entity_group") == "ANIMAL"
    }
    return sorted(animals)


def parse_args():
    parser = argparse.ArgumentParser(description="Run NER inference for animal extraction.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    animals = extract_animals(args.text, args.model_dir)
    print({"animals": animals})


if __name__ == "__main__":
    main()
