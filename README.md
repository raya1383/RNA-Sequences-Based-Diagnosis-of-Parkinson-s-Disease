# 🧬 RNA-Seq Based Parkinson’s Disease Diagnosis

### Feature Extraction Comparison using Machine Learning

This project implements a complete machine learning pipeline for **Parkinson’s disease diagnosis using RNA-seq gene expression data** and compares multiple feature selection methods to determine the most informative genes.

The goal is to evaluate which feature extraction method provides the best classification performance across several machine learning models.

---

# 📊 Project Overview

RNA-seq datasets are extremely high-dimensional (~40k genes) but contain relatively few samples (~73 patients).
This creates a **high-dimension, low-sample-size problem**, where feature selection plays a crucial role.

This notebook implements and compares several feature selection methods and evaluates them using multiple classifiers.

---

# 🧠 Feature Selection Methods Implemented

The following feature extraction techniques are implemented and compared:

### 1. Mean Difference Selection

Selects genes with the largest difference in mean expression between:

* Parkinson (PD)
* Control

Assumption: genes with larger expression differences are more discriminative.

---

### 2. Variance-Based Selection

Selects genes with the highest variance across samples.

Low-variance genes carry little information and may introduce noise.

---

### 3. Information Gain (IG)

Measures how much each gene reduces uncertainty about class labels.

Implemented using mutual information computed on the training set only to avoid data leakage.

---

### 4. Genetic Algorithm (GA)

Evolutionary optimization algorithm that searches for an optimal subset of genes.

Includes:

* population initialization
* crossover
* mutation
* fitness evaluation

A variance-based pre-filter is applied before GA to reduce computation time.

---

### 5. Wolf Search Algorithm (WSA)

A swarm-intelligence optimization method used to find informative gene subsets.

Steps:

* Pre-filter genes by variance
* Wolves search feature space
* Fitness evaluated using validation accuracy
* Best gene subset selected

---

# 🤖 Machine Learning Models Used

Each feature selection method is evaluated using:

* Decision Tree (DT)
* Support Vector Machine (SVM)
* Deep Neural Network (MLP-based DNN)
* XGBoost
* Logistic Regression (baseline)

All models are trained on training data and evaluated on a held-out test set.

---

# ⚙️ Pipeline

```text
Load RNA-seq dataset
        ↓
Preprocessing & normalization
        ↓
Train/Test split
        ↓
Feature Selection Methods
   - Mean difference
   - Variance
   - IG
   - GA
   - Wolf
        ↓
Model training
        ↓
Accuracy comparison
        ↓
Visualization
```

---

# 📈 Output

The notebook generates:

* Accuracy comparison table
* Bar chart comparing feature selection methods
* Best method per classifier
* Selected gene lists for each method

This allows direct evaluation of which feature selection strategy performs best.

---

# 🧪 Dataset

Dataset used:
**GSE68719 RNA-seq Parkinson dataset**

Contains:

* Parkinson patients
* Healthy controls
* ~40,000 gene expression features

---

# 📦 Requirements

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost torch
```

If using Apple Silicon:

```bash
pip install torch torchvision torchaudio
```

---

# 🚀 How to Run

1. Clone repository
2. Open Jupyter notebook
3. Run cells sequentially

```bash
git clone https://github.com/yourusername/parkinson-rnaseq-ml.git
cd parkinson-rnaseq-ml
jupyter notebook
```

---

# 🎯 Goal of This Project

This project aims to determine:

> Which feature extraction method produces the most informative gene subset for Parkinson’s disease classification?

The comparison is performed using identical training/testing conditions across all methods.
