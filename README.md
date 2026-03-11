# Winstars AI DS Internship Test

This repository contains both tasks from the internship assignment, each isolated in its own folder with dedicated documentation and dependency lists.

## Repository structure

```text
.
├── Task 1/
│   ├── README.md
│   ├── main.py
│   ├── requirements.txt
│   └── src/
├── docs/
│   ├── code_structure_guide.md
│   └── code_structure_guide.pdf
├── scripts/
│   └── generate_pdf.py
└── Task 2/
    ├── README.md
    ├── eda.ipynb
    ├── image_classification/
    ├── ner/
    ├── pipeline.py
    └── requirements.txt
```

## Task overview

### Task 1

Implements three MNIST classifiers behind a shared interface:

- Random Forest
- Feed-Forward Neural Network
- Convolutional Neural Network

### Task 2

Implements an NLP + Computer Vision pipeline:

- transformer-based NER for extracting animal entities from text;
- animal image classifier trained on a dataset with at least 10 classes;
- a pipeline script that compares both predictions and returns a boolean result.

## Notes

- All runnable details for each task are documented inside the corresponding task folder.
- The notebook in `Task 2` is focused on EDA and demo usage, while training and inference stay in parametrized `.py` scripts as required by the assignment.
- `docs/code_structure_guide.md` and its PDF version provide a compact walkthrough of the repository structure for faster review.
