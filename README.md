# Winstars-AI-DS-internship-test

## Task 1: Image Classification + OOP

This project implements three classification models for the MNIST dataset using object-oriented programming principles:

- Random Forest
- Feed-Forward Neural Network
- Convolutional Neural Network

### Structure and Approach

1. **Interface Design**
   - Created `MnistClassifierInterface` with two abstract methods: `train` and `predict`.
   - This ensures all models have a consistent API for training and prediction.

2. **Model Classes**
   - Implemented three separate classes:
     - `RandomForestClassifier` (uses scikit-learn)
     - `FeedForwardNNClassifier` (uses TensorFlow/Keras)
     - `CNNClassifier` (uses TensorFlow/Keras)
   - Each class inherits from `MnistClassifierInterface` and implements its own logic for training and prediction.

3. **Unified Wrapper**
   - Created `MnistClassifier` class, which takes an algorithm name ('rf', 'nn', 'cnn') as input and delegates all operations to the corresponding model class.
   - This allows switching between models with a single interface, making code reusable and easy to test.

4. **Data Loading and Testing**
   - Used TensorFlow's built-in MNIST loader for data preparation.
   - Provided a `main.py` script to test all models and print their accuracy.

### Why This Structure?
- **OOP Principles:** Clear separation of interface and implementation, easy to extend with new models.
- **Reusability:** The wrapper allows using any model with the same input/output structure.
- **Maintainability:** Each model is isolated, making debugging and improvements straightforward.
- **Scalability:** New algorithms can be added by simply implementing the interface and updating the wrapper.

### How to Run Task 1
1. Install dependencies:
   ```bash
   pip install scikit-learn tensorflow numpy
   ```
2. Run the main script:
   ```bash
   python Task\ 1/main.py
   ```

### Task 1 Structure
```
Task 1/
├── src/
│   ├── mnist_classifier_interface.py
│   ├── random_forest_classifier.py
│   ├── feed_forward_nn_classifier.py
│   ├── cnn_classifier.py
│   ├── mnist_classifier.py
├── main.py
├── README.md
```

---

## Task 2: Named Entity Recognition + Image Classification

This project builds an ML pipeline combining NLP (NER) and Computer Vision (image classification) to verify user statements about animals in images.

### Dataset
- The animal classification dataset should contain at least 10 distinct animal classes (e.g., cat, dog, cow, horse, sheep, lion, tiger, bear, fox, wolf).
- Each class must have a sufficient number of images for training and evaluation.
- Dataset structure: each class in a separate folder, images named arbitrarily.

### Exploratory Data Analysis (EDA)
- The EDA notebook visualizes class distribution, sample images, and basic statistics.
- Helps identify class imbalance and dataset quality.

### NER Model
- Uses a transformer-based model (e.g., BERT, DistilBERT) for extracting animal names from text.
- Training data: sentences containing animal names, annotated in BIO format.
- Scripts provided for training and inference.
- Model outputs entity labels for each token, allowing extraction of animal names from user input.

### Image Classification Model
- Uses a neural network or CNN (e.g., TensorFlow/Keras) for animal image classification.
- Images are resized, normalized, and split into train/validation/test sets.
- Scripts provided for training and inference.
- Model predicts the animal class for a given image.

### Pipeline Script
- Accepts two inputs: a text message and an image.
- Steps:
  1. Extract animal entity from text using NER model.
  2. Classify the animal in the image using the image model.
  3. Compare results and output a boolean (True if text matches image, False otherwise).
- Handles flexible user input (e.g., "A lion is in the photo", "I see a dog", etc.).

### Example Usage
- User provides: "There is a cat in the picture." and an image of a cat.
- Pipeline extracts "cat" from text, classifies image as "cat", returns True.
- If image is a dog, returns False.

### Edge Cases
- Handles synonyms and plural forms (e.g., "cats", "dogs").
- Handles sentences with multiple animals (returns True if any match).
- Handles images with multiple animals (advanced: multi-label classification).

### Folder Structure
```
Task 2/
├── ner/                  # NER scripts and models
├── image_classification/ # Image classification scripts and models
├── eda.ipynb             # Jupyter notebook for EDA and demo
├── pipeline.py           # Pipeline script (text + image → boolean)
├── README.md             # Task 2 documentation
├── requirements.txt      # Dependencies
```

---

For more details, see the EDA notebook and script documentation in each folder.

Both tasks are documented and structured for clarity, reproducibility, and easy setup. All code and instructions are in English.