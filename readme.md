# DecodeLabs Project 2: Data Classification Using AI
## AI Engineering Industrial Training Kit (Batch 2026)

This repository contains the professional, production-grade implementation of **Project 2: Data Classification Using AI** for the DecodeLabs internship curriculum. The objective of this project is to build an end-to-end Machine Learning pipeline to train, tune, validate, and deploy a K-Nearest Neighbors (KNN) classification model on the classic **Iris Benchmark Dataset**.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Modular Architecture](#modular-architecture)
3. [Algorithmic Methodology](#algorithmic-methodology)
4. [Installation & Setup](#installation--setup)
5. [Usage Instructions](#usage-instructions)
6. [Testing & Verification](#testing--verification)
7. [Artifacts & Results](#artifacts--results)

---

## 🔍 Project Overview

The project steps beyond simple rule-based heuristics into **Supervised Machine Learning**. We utilize the physical dimensions of Iris flowers (Sepal Length, Sepal Width, Petal Length, and Petal Width) to classify each sample into one of three species:
- **Setosa**
- **Versicolor**
- **Virginica**

Our architecture leverages a modular code design following professional software engineering standards, separating data ingestion, processing, model definition, optimization, evaluation, and interface layers.

---

## 🛠️ Modular Architecture

The repository is structured as follows:

```
project 2/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py   # Ingests raw data, performs stratified train-test splits, scales features
│   ├── model.py         # Encapsulates KNeighborsClassifier with fit/predict wrappers and serialization
│   ├── tuner.py         # Automates K hyperparameter tuning via cross-validation and elbow-plots
│   ├── metrics.py       # Evaluates model performance and renders Confusion Matrices
│   ├── pipeline.py      # Main pipeline orchestration script
│   └── cli.py           # Command-Line Interface for training, parameter sweeps, and predictions
│
├── notebooks/
│   └── iris_exploration.ipynb  # Interactive Jupyter notebook walkthrough (EDA, scaling, tuning)
│
├── tests/
│   └── test_pipeline.py # Comprehensive unit test suite
│
├── build/               # Generated automatically upon pipeline execution
│   ├── model.joblib     # Serialized trained KNN model
│   ├── scaler.joblib    # Serialized fitted StandardScaler object
│   ├── elbow_plot.png   # Visualization of K error-rates curve
│   └── confusion_matrix.png # Diagnostic evaluation heatmap
│
├── requirements.txt     # Python package requirements
├── Artificial intelligence P2.pdf # Project instructions
└── readme.md            # Comprehensive project guide
```

---

## 🧠 Algorithmic Methodology

Our pipeline is built on the **IPO Framework (Input -> Process -> Output)**:

### 1. Input: Feature Scaling (Slide 9)
Raw numerical measurements in spatial-distance classifiers (like KNN) can suffer from feature magnitude bias. We apply a **StandardScaler** to map features to:
- **Mean = 0**
- **Variance = 1**

### 2. Process: Shuffled Splitting & KNN Tuning (Slides 10, 11, 12, 13)
- **Train-Test Split**: To secure structural validation integrity, we split the data into **80% Training Set** (for pattern recognition) and **20% Test Set** (for validation). This split is randomized/shuffled to remove ordering bias and stratified to preserve equal class distribution.
- **K-Nearest Neighbors**: KNN classifies an unlabelled data point based on the majority vote of its $K$ nearest neighbors using Euclidean distance.
- **Hyperparameter Tuning**: We perform 5-fold cross-validation across $K \in [1, 25]$ to locate the "Elbow" representing the optimal balance between overfitting ($K=1$, high noise) and underfitting (large $K$, high bias). We prefer odd values of $K$ to prevent voting ties.

### 3. Output: Performance Diagnostics (Slides 14, 15, 16, 17)
Relying strictly on classification accuracy can be misleading (the "Accuracy Mirage"). We generate full diagnostic reports:
- **Accuracy**: Overall fraction of correct predictions.
- **Precision**: Measure of exactness (minimizing false alarms).
- **Recall**: Measure of completeness (minimizing missed detections).
- **F1-Score**: The harmonic mean of Precision and Recall.
- **Confusion Matrix**: A diagnostic grid showing actual vs. predicted classifications.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11 or higher
- `pip` package manager

### Steps
1. Navigate to the project root directory:
   ```bash
   cd "d:\Honey\DecodeLabs\project 2"
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Instructions

We provide three different interfaces to interact with the project:

### 1. Interactive Command-Line Interface (CLI)
Run the user-friendly command-line utility to run training runs, tune parameters, or perform real-time predictions:
```bash
python src/cli.py
```
**CLI Options:**
1. *Run Complete Training & Preprocessing Pipeline*: Loads data, runs parameter search, trains model with optimal $K$, evaluates performance, and outputs model, scaler, and graphs to the `build/` directory.
2. *Predict Iris Species*: Renders predictions on a custom flower's features based on the saved `model.joblib` and `scaler.joblib`.
3. *View K-Parameter Hyperparameter Tuning Details*: Displays tabular error rates for each K neighbor sweep.

### 2. Experimental Jupyter Notebook
For interactive exploratory data analysis, visualizations, and research, run the Jupyter Notebook:
```bash
jupyter notebook notebooks/iris_exploration.ipynb
```

### 3. Direct Pipeline Invocation
To run the automated machine learning pipeline programmatically as part of a script:
```python
from src.pipeline import run_pipeline
results = run_pipeline(output_dir='build')
print("Model Accuracy:", results['metrics']['accuracy'])
```

---

## 🧪 Testing & Verification

A robust unit test suite has been built under `tests/` using Python's standard `unittest` library. It validates:
- Correct data load shape and labels.
- Balanced stratified splitting ratios.
- Standard scaler calculations (means and variances scaling).
- KNN wrapper fit/predict mechanisms and class probabilities output.
- Parameters search range checks.
- Classification metrics computations.

Run the tests with:
```bash
python -m unittest discover -s tests
```

---

## 📊 Artifacts & Results

Upon running the pipeline, the following artifacts are generated in the `build/` directory:
- **`model.joblib`**: The serialized KNN classifier.
- **`scaler.joblib`**: The serialized scaler parameters (for centering new inputs).
- **`elbow_plot.png`**: Plot depicting error rates vs $K$ to justify model configuration.
- **`confusion_matrix.png`**: Heatmap displaying final model validation diagnostics.
