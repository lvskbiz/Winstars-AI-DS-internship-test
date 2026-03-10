# Task 1: Image Classification + OOP

This task implements three MNIST classifiers behind the same interface:

- `rf`: Random Forest based on scikit-learn
- `nn`: Feed-Forward Neural Network based on TensorFlow/Keras
- `cnn`: Convolutional Neural Network based on TensorFlow/Keras

## Project structure

```text
Task 1/
├── main.py
├── requirements.txt
└── src/
    ├── cnn_classifier.py
    ├── feed_forward_nn_classifier.py
    ├── mnist_classifier.py
    ├── mnist_classifier_interface.py
    └── random_forest_classifier.py
```

## Design

- `MnistClassifierInterface` defines the common `train` and `predict` API.
- Each model class implements the same interface.
- `MnistClassifier` hides the implementation details and dispatches by algorithm name.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r "Task 1/requirements.txt"
```

## Run

Train and evaluate a single model:

```bash
python3 "Task 1/main.py" --algorithm rf --sample-size 5000
python3 "Task 1/main.py" --algorithm nn --epochs 3 --sample-size 10000
python3 "Task 1/main.py" --algorithm cnn --epochs 3 --sample-size 10000
```

`main.py` first tries to load MNIST from `tensorflow.keras.datasets`. If TensorFlow dataset loading is unavailable, it falls back to `fetch_openml("mnist_784")`.

For offline smoke tests in restricted environments, you can also use:

```bash
python3 "Task 1/main.py" --algorithm rf --dataset-source digits --sample-size 500
```

The `digits` option is only a local fallback for verification. The target dataset for the assignment remains MNIST.

## Quick validation

The repository was smoke-tested locally with:

```bash
python3 "Task 1/main.py" --algorithm rf --dataset-source keras --sample-size 5000
```

Observed result on the sampled MNIST subset:

- `rf` test accuracy: `0.9340`
