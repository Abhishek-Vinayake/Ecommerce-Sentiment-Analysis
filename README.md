# Ecommerce Sentiment Analysis: Classical ML vs Transformer Benchmark

## Overview

This project benchmarks classical machine learning and transformer-based NLP models for large-scale ecommerce sentiment analysis.

The project compares:

- TF-IDF + LinearSVC
- DistilBERT

on a cleaned ecommerce review dataset containing more than 900,000 reviews labeled as:

- positive
- negative
- neutral

The goal was to evaluate:
- classification performance
- inference efficiency
- deployment tradeoffs
- transformer vs classical ML effectiveness on structured review data

---

# Dataset

## Dataset Size

| Metric | Value |
|---|---|
| Total Reviews | 914,351 |
| Unique Reviews | 914,351 |
| Missing Values | 0 |
| Duplicate Rows | 0 |

## Label Distribution

| Sentiment | Count |
|---|---|
| Positive | 460,660 |
| Negative | 365,285 |
| Neutral | 88,406 |

## Dataset Features

| Column | Description |
|---|---|
| text | Cleaned ecommerce review text |
| sentiment | Sentiment label |

---

# Preprocessing

The dataset was created by combining and standardizing multiple ecommerce sentiment datasets.

Preprocessing steps included:

- lowercasing
- text normalization
- duplicate removal
- null removal
- unified sentiment labels
- schema standardization
- dataset merging
- cleaning noisy review text

---

# Models Compared

## 1. TF-IDF + LinearSVC

### Pipeline
- TF-IDF Vectorization
- Linear Support Vector Classifier

### Advantages
- fast inference
- lightweight deployment
- strong sparse-text performance
- efficient on large datasets

---

## 2. DistilBERT

### Pipeline
- HuggingFace Transformers
- DistilBERT sequence classification

### Advantages
- contextual embeddings
- transformer-based NLP
- semantic understanding capability

---

# Benchmark Results

Evaluation performed on:
- 10% random dataset sample
- ~91,000 review samples
- GPU-enabled transformer inference

## Model Comparison

| Metric | SVC + TFIDF | DistilBERT |
|---|---|---|
| Accuracy | 89.87% | 87.01% |
| Precision | 90.82% | 86.84% |
| Recall | 89.87% | 87.01% |
| F1 Score | 90.23% | 86.84% |

---

# Inference Performance

| Model | Inference Time |
|---|---|
| SVC + TFIDF | ~34 sec |
| DistilBERT | ~576 sec |

DistilBERT inference was significantly slower despite GPU acceleration.

---

# Key Findings

## 1. Classical ML Outperformed DistilBERT

TF-IDF + LinearSVC achieved:
- higher F1-score
- higher accuracy
- faster inference

compared to DistilBERT on this dataset.

---

## 2. Neutral Class Was Challenging

The transformer model struggled significantly with neutral sentiment classification.

SVC demonstrated better neutral-class separation and overall class balance handling.

---

## 3. Classical ML Remains Strong for Structured NLP

This benchmark demonstrates that properly engineered classical NLP pipelines can still outperform transformer architectures on structured sentiment analysis datasets.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- HuggingFace Transformers
- PyTorch
- Matplotlib
- Joblib

---

# Project Structure

```text
project/
│
├── dataset/
│   └── sentiment_final_cleaned.csv
│
├── models/
│   ├── SVC-TFIDF/
│   └── DistilBert/
│
├── notebooks/
│   ├── sentiment_analysis_tfidf_svc.ipynb
│   ├── sentiment_analysis_distilbert.ipynb
│   └── model_comparison.ipynb
│
└── README.md
```

---

# Future Improvements

Potential future experiments:

- RoBERTa benchmarking
- class-weighted transformer training
- focal loss
- balanced sampling
- FastAPI deployment
- Docker containerization
- real-time inference API

---

# Kaggle Dataset

Dataset published on Kaggle:

[dataset link](https://www.kaggle.com/datasets/abhishekvinayake/ecommerce-sentiment-dataset-900k)

---

# Conclusion

This project demonstrates that classical machine learning models remain highly competitive for large-scale sentiment analysis tasks when combined with strong preprocessing and feature engineering.

The benchmark also highlights the importance of evaluating:
- performance
- efficiency
- deployment cost
- inference speed

rather than assuming transformer superiority by default.

