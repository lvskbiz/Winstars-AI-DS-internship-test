from __future__ import annotations

import argparse
import os
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.mnist_classifier import MnistClassifier


def load_mnist(dataset_source: str):
    """Load MNIST with optional offline fallback for smoke tests."""
    if dataset_source == "digits":
        from sklearn.datasets import load_digits

        digits = load_digits()
        X = digits.images.astype("float32")
        y = digits.target.astype("int64")
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    local_mnist_path = PROJECT_ROOT / ".cache" / "mnist.npz"
    if local_mnist_path.exists() and dataset_source in {"auto", "keras"}:
        with np.load(local_mnist_path) as data:
            X_train = data["x_train"]
            y_train = data["y_train"]
            X_test = data["x_test"]
            y_test = data["y_test"]
        return X_train, X_test, y_train, y_test

    if dataset_source in {"auto", "keras"}:
        try:
            mnist = import_module("tensorflow.keras.datasets").mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            return X_train, X_test, y_train, y_test
        except Exception:
            if dataset_source == "keras":
                raise

    from sklearn.datasets import fetch_openml

    cache_dir = PROJECT_ROOT / ".cache" / "sklearn_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        return_X_y=True,
        data_home=str(cache_dir),
    )
    X = X.astype("float32").reshape(-1, 28, 28)
    y = y.astype("int64")
    return train_test_split(X, y, test_size=10000, random_state=42, stratify=y)


def prepare_features(algorithm: str, X_train, X_test):
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    if algorithm == "rf":
        return X_train.reshape(len(X_train), -1), X_test.reshape(len(X_test), -1)

    if algorithm == "cnn":
        return X_train[..., np.newaxis], X_test[..., np.newaxis]

    return X_train, X_test


def apply_sample_limit(X_train, X_test, y_train, y_test, sample_size: int | None):
    if sample_size is None:
        return X_train, X_test, y_train, y_test

    test_size = max(1000, sample_size // 5)
    return (
        X_train[:sample_size],
        X_test[:test_size],
        y_train[:sample_size],
        y_test[:test_size],
    )


def run_experiment(
    algorithm: str,
    epochs: int,
    batch_size: int,
    sample_size: int | None,
    dataset_source: str,
):
    X_train, X_test, y_train, y_test = load_mnist(dataset_source)
    X_train, X_test, y_train, y_test = apply_sample_limit(
        X_train, X_test, y_train, y_test, sample_size
    )

    X_train, X_test = prepare_features(algorithm, X_train, X_test)
    classifier = MnistClassifier(algorithm)

    train_kwargs = {}
    if algorithm in {"nn", "cnn"}:
        train_kwargs["epochs"] = epochs
        train_kwargs["batch_size"] = batch_size

    classifier.train(X_train, y_train, **train_kwargs)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Algorithm: {algorithm}")
    print(f"Test accuracy: {accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate MNIST classifiers.")
    parser.add_argument("--algorithm", choices=["rf", "nn", "cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional subset of the training set for faster smoke tests.",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["auto", "keras", "openml", "digits"],
        default="auto",
        help="Use digits only for offline smoke tests; the assignment target remains MNIST.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        algorithm=args.algorithm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        dataset_source=args.dataset_source,
    )
