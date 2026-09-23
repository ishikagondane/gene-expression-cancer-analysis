# Gene Expression Cancer RNA-Seq Analysis

## Overview

This project analyzes **Gene Expression Cancer RNA-Seq data** using machine learning techniques for **clustering and classification**.

The dataset contains gene expression profiles from five cancer types:

* BRCA
* KIRC
* COAD
* LUAD
* PRAD

## Methods

### Clustering

* **K-Means Clustering**
* **Hierarchical Agglomerative Clustering**
* **PCA** for 2D visualization
* **Dendrogram** for hierarchical clustering

### Classification

* **Random Forest Classifier**
* 80% training and 20% testing split
* Evaluation using **accuracy, precision, recall, and F1-score**

## Libraries

```text
pandas
numpy
matplotlib
scikit-learn
scipy
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

## Dataset

The dataset was obtained from **Kaggle** and consists of gene expression data (`data.csv`) and cancer-type labels (`labels.csv`).

## Project Structure

```text
Machine-learning/
│
├── classification-clustering.py
└── README.md
```

## Objective

The objective is to explore patterns in gene expression data through **unsupervised clustering** and classify samples into their corresponding **cancer types using Random Forest**.
