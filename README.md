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

### Solution Structure
- Jupyter notebook for exploratory data analysis (EDA)
- Parametrized train/inference scripts for NER (transformer-based model)
- Parametrized train/inference scripts for Image Classification
- Python script for the pipeline (takes text and image, outputs boolean)

### Approach
1. **Dataset**: Animal classification dataset with at least 10 classes.
2. **NER Model**: Transformer-based model (e.g., BERT) trained to extract animal names from text.
3. **Image Classification Model**: Neural network/CNN trained to classify animal images.
4. **Pipeline**: Accepts text and image, extracts animal entity from text, classifies image, and checks if they match.

### How to Run Task 2
1. Install dependencies:
   ```bash
   pip install -r Task\ 2/requirements.txt
   ```
2. Run EDA notebook:
   - Open `Task 2/eda.ipynb` in Jupyter.
3. Train/infer NER and image models using provided scripts.
4. Run pipeline script with text and image inputs.

### Task 2 Structure
```
Task 2/
├── ner/
├── image_classification/
├── eda.ipynb
├── README.md
├── requirements.txt
```

---

Both tasks are documented and structured for clarity, reproducibility, and easy setup. All code and instructions are in English.