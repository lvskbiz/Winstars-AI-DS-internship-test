from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train an animal image classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root folder with train/valid subfolders.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def build_dataloaders(data_dir: Path, image_size: int, batch_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(data_dir / "valid", transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, train_loader, valid_loader


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_loader, valid_loader = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size
    )

    weights = models.ResNet18_Weights.DEFAULT if args.pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        valid_accuracy = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch + 1}: valid_accuracy={valid_accuracy:.4f}")
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), args.output_dir / "model.pt")

    with (args.output_dir / "class_names.json").open("w", encoding="utf-8") as handle:
        json.dump(train_dataset.classes, handle, ensure_ascii=True, indent=2)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_valid_accuracy": best_accuracy}, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
