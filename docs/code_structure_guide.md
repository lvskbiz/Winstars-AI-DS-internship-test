# Winstars AI DS Internship Test

## Guide to the Code Structure of Task 1 and Task 2

This document is a practical map of the repository. It is written to help a reader quickly understand:

- where the entry points are;
- how data moves through each task;
- which files own which responsibilities;
- what the important design decisions and limitations are.

---

## 1. Repository Layout

The repository is split into two independent tasks:

- `Task 1/` implements OOP wrappers around three MNIST classifiers.
- `Task 2/` implements a two-model pipeline:
  one model extracts animal names from text, another classifies the animal in an image, and a pipeline compares the two outputs.

Top-level files:

- `README.md`: short overview of both tasks.
- `description.txt`: original internship assignment.

---

## 2. Task 1 Overview

### Goal

Task 1 solves MNIST digit classification with three interchangeable algorithms:

- Random Forest (`rf`)
- Feed-Forward Neural Network (`nn`)
- Convolutional Neural Network (`cnn`)

The key requirement is a shared interface, so the outer API stays the same regardless of the chosen model.

### Main Flow

Runtime flow in Task 1:

1. `Task 1/main.py` parses CLI arguments.
2. It loads MNIST data or a fallback dataset.
3. It reshapes data depending on the selected algorithm.
4. It creates `MnistClassifier(algorithm=...)`.
5. The wrapper instantiates the correct concrete model.
6. `train(...)` is called through the shared API.
7. `predict(...)` is called through the same API.
8. Accuracy is computed and printed.

### Task 1 File Responsibilities

#### `Task 1/main.py`

This is the entry point for the whole task.

Responsibilities:

- parse command-line flags such as `--algorithm`, `--epochs`, `--batch-size`;
- load the dataset with fallback logic;
- normalize and reshape features according to the selected model;
- run training and evaluation;
- print final test accuracy.

Important details:

- `load_mnist(...)` first tries a local cache in `Task 1/.cache/mnist.npz`.
- If needed, it tries `tensorflow.keras.datasets.mnist`.
- If that fails, it falls back to `fetch_openml("mnist_784")`.
- For offline smoke tests, it can also use `sklearn.datasets.load_digits` via `--dataset-source digits`.

This means `main.py` is not just a launcher; it also owns dataset acquisition and preprocessing strategy.

#### `Task 1/src/mnist_classifier_interface.py`

Defines the abstract contract:

- `train(X_train, y_train, **kwargs)`
- `predict(X_test)`

This file exists to enforce a stable API across all model implementations.

#### `Task 1/src/mnist_classifier.py`

This is the factory/wrapper layer.

Responsibilities:

- map short algorithm names to classes;
- hide implementation details from the caller;
- expose one consistent public API.

The user of this class does not need to know whether the underlying model is scikit-learn or TensorFlow based.

#### `Task 1/src/random_forest_classifier.py`

Implements the `rf` option using scikit-learn.

Responsibilities:

- initialize `sklearn.ensemble.RandomForestClassifier`;
- fit on flattened image vectors;
- return class predictions.

Design implication:

- this model expects tabular input, so images are flattened from `28x28` into a 784-dimensional vector.

#### `Task 1/src/feed_forward_nn_classifier.py`

Implements the `nn` option using TensorFlow/Keras.

Architecture:

- `Input(28, 28)`
- `Flatten`
- `Dense(256, relu)`
- `Dropout(0.2)`
- `Dense(128, relu)`
- `Dense(10, softmax)`

Responsibilities:

- build and compile the dense neural network;
- train with `epochs` and `batch_size`;
- convert softmax probabilities into class indices using `argmax`.

Design implication:

- unlike Random Forest, the model still receives a 2D image tensor, but it flattens internally.

#### `Task 1/src/cnn_classifier.py`

Implements the `cnn` option using TensorFlow/Keras.

Architecture:

- `Input(28, 28, 1)`
- `Conv2D(32)`
- `MaxPool2D`
- `Conv2D(64)`
- `MaxPool2D`
- `Flatten`
- `Dense(128, relu)`
- `Dropout(0.3)`
- `Dense(10, softmax)`

Responsibilities:

- build a CNN specialized for image inputs;
- train and predict through the same interface as the other models.

Design implication:

- this model requires a channel dimension, so `main.py` converts the input into shape `(N, 28, 28, 1)`.

### How Preprocessing Depends on Algorithm

One of the most important details in Task 1 is `prepare_features(...)` in `main.py`.

- For `rf`: images are normalized and flattened.
- For `nn`: images are normalized but kept as `28x28`.
- For `cnn`: images are normalized and expanded with a channel dimension.

This is the main reason the wrapper pattern works here:
the caller sees one API, while shape handling is hidden behind the task logic.

### Task 1 Design Strengths

- Clear separation between interface, wrapper, and concrete implementations.
- Consistent training/prediction API for all models.
- Practical fallback logic for restricted or offline environments.
- Minimal but valid demonstration of OOP polymorphism.

### Task 1 Limitations

- No model saving/loading layer.
- No separate evaluation module; evaluation stays in `main.py`.
- Dataset loading and experiment running are mixed in the same file.
- Neural-network hyperparameters are fixed in code rather than configurable via constructor arguments.

---

## 3. Task 2 Overview

### Goal

Task 2 combines NLP and computer vision:

- the NER model extracts animal entities from text;
- the image classifier predicts the animal shown in an image;
- `pipeline.py` returns `True` if the image prediction matches one of the extracted animals, otherwise `False`.

### End-to-End Flow

The complete pipeline follows this logic:

1. Prepare NER data with `Task 2/ner/prepare_data.py`.
2. Prepare image data with `Task 2/image_classification/prepare_data.py`.
3. Train the NER model with `Task 2/ner/train.py`.
4. Train the image classifier with `Task 2/image_classification/train.py`.
5. Run `Task 2/pipeline.py` with text, image, and both model directories.
6. The pipeline calls NER inference and image inference.
7. It compares the canonicalized animal names and returns a boolean.

Task 2 is therefore a composition task rather than a single-model task.

---

## 4. Task 2 NER Module

Folder:

- `Task 2/ner/`

### `Task 2/ner/prepare_data.py`

This script generates synthetic training data in JSONL format.

Main idea:

- define canonical animals and alias forms;
- insert them into sentence templates;
- generate BIO tags (`B-ANIMAL`, `I-ANIMAL`, `O`);
- add negative examples with no animal entity;
- split into train/validation/test.

Important internal pieces:

- `ANIMALS`: canonical label to alias list mapping.
- `TEMPLATES`: single-animal sentence patterns.
- `MULTI_TEMPLATES`: sentences with two animals.
- `NEGATIVE_TEMPLATES`: sentences with no entities.
- `to_bio(...)`: converts token spans into BIO labels.

What this file really does architecturally:

- it removes the need for a manually annotated dataset;
- it makes the NER training fully reproducible;
- it constrains the problem to a template-based language space.

Practical consequence:

- the model can learn the target label format easily,
  but generalization to more natural and messy user text may be limited.

### `Task 2/ner/train.py`

This is the NER training entry point.

Responsibilities:

- load JSONL data;
- convert it into Hugging Face `Dataset` objects;
- tokenize split tokens with `is_split_into_words=True`;
- align word-level BIO labels to subword tokens;
- fine-tune a transformer token-classification model;
- save both model and tokenizer.

Key design decisions:

- default base model: `distilbert/distilbert-base-uncased`;
- labels are fixed to `["O", "B-ANIMAL", "I-ANIMAL"]`;
- special tokens and non-leading subtokens receive label `-100`,
  so they are ignored in the loss.

This file is the most important implementation detail in the NER pipeline, because label alignment is the critical step when moving from word-level annotations to transformer tokenization.

### `Task 2/ner/inference.py`

This file serves as the NER inference layer.

Responsibilities:

- load the trained model with Hugging Face `pipeline(task="token-classification")`;
- aggregate subword predictions using `aggregation_strategy="simple"`;
- keep only entities with group `ANIMAL`;
- normalize plural or variant surface forms into canonical labels.

Key point:

- the normalization dictionary is not identical to the training aliases.
  For example, it includes labels like `lion`, `tiger`, `bear`, `fox`, `wolf`,
  even though the synthetic training data generator focuses on another animal set.

This means inference tries to be broader than the training generator, but the trained model may not reliably support those extra forms unless it generalizes beyond the synthetic data.

### NER Data Contract

Every JSONL row must look like:

```json
{"tokens": ["There", "is", "a", "cow", "."], "ner_tags": ["O", "O", "O", "B-ANIMAL", "O"]}
```

This contract is important because both training and label alignment assume a tokenized input rather than raw text with character offsets.

---

## 5. Task 2 Image Classification Module

Folder:

- `Task 2/image_classification/`

### `Task 2/image_classification/prepare_data.py`

This script downloads and restructures the animal image dataset.

Responsibilities:

- load `Rapidata/Animals-10` through Hugging Face Datasets;
- convert source labels into English class names using `CLASS_MAP`;
- split each class into `train`, `valid`, and `test`;
- save images into `ImageFolder`-compatible directories.

Why this matters:

- the training script uses `torchvision.datasets.ImageFolder`,
  so the dataset must exist as folders grouped by class name.

This file therefore acts as the bridge between a Hugging Face dataset and a PyTorch training pipeline.

### `Task 2/image_classification/train.py`

This is the image-model training entry point.

Responsibilities:

- build transforms and dataloaders;
- create a `resnet18` model;
- optionally initialize it with pretrained weights;
- replace the final classification layer with the correct number of classes;
- train with cross-entropy loss;
- evaluate on validation data after each epoch;
- save the best checkpoint and class names.

Core components:

- `build_dataloaders(...)`
- `evaluate(...)`
- `main()`

Artifacts written to `output-dir`:

- `model.pt`: best model weights;
- `class_names.json`: class index to class name mapping;
- `metrics.json`: best validation accuracy.

Important design note:

- inference depends on `class_names.json` and `model.pt`,
  so these two files are part of the formal output contract of training.

### `Task 2/image_classification/inference.py`

This file performs single-image inference.

Responsibilities:

- load class names from JSON;
- rebuild the same ResNet-18 head size;
- load saved weights;
- preprocess one image;
- return the predicted class label.

Important technical observation:

- inference always resizes to `224x224`,
  which matches the training default but is hard-coded here.
- if training were done with another `--image-size`, inference would no longer exactly match training preprocessing.

That is a small coupling risk worth remembering.

---

## 6. Task 2 Pipeline Orchestration

### `Task 2/pipeline.py`

This is the orchestration layer that combines both trained models.

Responsibilities:

- accept text and image input paths from CLI;
- call `extract_animals(...)` from the NER module;
- call `predict_image(...)` from the image module;
- return `True` if the predicted image class is mentioned in text.

Core logic:

- `extracted_animals = extract_animals(text, ner_model_dir)`
- `predicted_animal = predict_image(image_path, image_model_dir)`
- `return predicted_animal in extracted_animals`

This is intentionally simple. The pipeline is not doing probabilistic reasoning or ranking; it performs exact membership matching after normalization.

### Pipeline Behavior in Edge Cases

- If text contains multiple animals, the pipeline returns `True` if the image class matches any extracted one.
- If NER extracts nothing, the result is `False`.
- If the image model predicts the wrong class, the result is `False` even if the text is correct.
- If the text uses animal wording outside the NER model's effective vocabulary, the result may be `False` due to failed extraction rather than failed image recognition.

This is important when interpreting pipeline failures:
the error can come from either submodel.

---

## 7. The Role of `eda.ipynb`

File:

- `Task 2/eda.ipynb`

The notebook is not the core implementation layer. Its role is supporting analysis and demonstration.

From the notebook structure, it includes:

- a dataset EDA section;
- commands and examples for training/inference;
- a note that the full pipeline should return one boolean output.

So the production logic lives in the `.py` scripts,
while the notebook serves as the demonstration artifact requested by the assignment.

---

## 8. Dependency Split

### Task 1 dependencies

- `numpy`
- `scikit-learn`
- `tensorflow`

Interpretation:

- scikit-learn powers the Random Forest and evaluation helpers;
- TensorFlow powers both neural-network classifiers and possibly MNIST loading.

### Task 2 dependencies

- `accelerate`
- `datasets`
- `matplotlib`
- `pandas`
- `pillow`
- `scikit-learn`
- `torch`
- `torchvision`
- `transformers`

Interpretation:

- Hugging Face stack is used for NER and dataset loading;
- PyTorch/Torchvision power the image model;
- Pillow is used for image IO during inference;
- notebook/EDA support likely uses pandas and matplotlib.

---

## 9. Architectural Comparison Between the Two Tasks

### Task 1

Task 1 is centered on OOP abstraction.

Its core question is:
how to provide one stable classifier API across multiple algorithm implementations?

The design answer is:

- one abstract interface;
- three concrete implementations;
- one wrapper/factory to select the implementation.

### Task 2

Task 2 is centered on pipeline composition.

Its core question is:
how to combine two different ML systems so one interprets text and another validates an image?

The design answer is:

- separate modules for data prep, training, and inference;
- clear artifact outputs from training;
- one thin orchestration script that combines the inference outputs.

In short:

- Task 1 emphasizes polymorphism.
- Task 2 emphasizes modular ML workflow composition.

---

## 10. Most Important Files to Read First

If you want to understand the repository quickly, read in this order:

### For Task 1

1. `Task 1/main.py`
2. `Task 1/src/mnist_classifier.py`
3. `Task 1/src/mnist_classifier_interface.py`
4. `Task 1/src/random_forest_classifier.py`
5. `Task 1/src/feed_forward_nn_classifier.py`
6. `Task 1/src/cnn_classifier.py`

Reason:

- `main.py` explains the runtime flow;
- the wrapper and interface explain the architecture;
- the model files explain implementation differences.

### For Task 2

1. `Task 2/pipeline.py`
2. `Task 2/ner/inference.py`
3. `Task 2/image_classification/inference.py`
4. `Task 2/ner/train.py`
5. `Task 2/image_classification/train.py`
6. `Task 2/ner/prepare_data.py`
7. `Task 2/image_classification/prepare_data.py`

Reason:

- the pipeline shows the final business logic first;
- inference files show what inputs/outputs each submodel exposes;
- training and data-prep files explain where those artifacts come from.

---

## 11. Final Summary

The repository is well-separated by task and reasonably easy to follow once the entry points are known.

The most important structural idea in Task 1 is the shared classifier interface with a wrapper that hides model-specific implementations.

The most important structural idea in Task 2 is the split into:

- data preparation;
- model training;
- model inference;
- final orchestration.

If you remember only one sentence for each task, use these:

- Task 1: one API, three MNIST model implementations.
- Task 2: one NER model plus one image model combined into a boolean decision pipeline.
