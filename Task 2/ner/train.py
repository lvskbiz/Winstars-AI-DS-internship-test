from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

LABELS = ["O", "B-ANIMAL", "I-ANIMAL"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer NER model for animal extraction.")
    parser.add_argument("--train-file", type=Path, required=True, help="JSONL file with tokens and ner_tags fields.")
    parser.add_argument("--valid-file", type=Path, required=True, help="Validation JSONL file with tokens and ner_tags fields.")
    parser.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align_labels(batch, tokenizer, label_to_id):
    tokenized = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    labels = []
    for sample_index, word_labels in enumerate(batch["ner_tags"]):
        previous_word_id = None
        label_ids = []
        for word_id in tokenized.word_ids(batch_index=sample_index):
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label_to_id[word_labels[word_id]])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


def tokenize_dataset(dataset, tokenizer, label_to_id):
    return dataset.map(
        align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id},
        remove_columns=dataset.column_names,
    )


def main():
    args = parse_args()
    label_to_id = {label: index for index, label in enumerate(LABELS)}

    train_dataset = Dataset.from_list(load_jsonl(args.train_file))
    valid_dataset = Dataset.from_list(load_jsonl(args.valid_file))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_train = tokenize_dataset(train_dataset, tokenizer, label_to_id)
    tokenized_valid = tokenize_dataset(valid_dataset, tokenizer, label_to_id)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=label_to_id,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
