# Project Context: Anomaly Detection in Encrypted Network Traffic

## Project Overview
This is a university dissertation project (University of Warwick) investigating machine learning-based anomaly detection in encrypted network traffic without payload inspection.

## Core Objective
Detect anomalous behaviour in encrypted network traffic using **flow-level statistical features** and **unsupervised machine learning**, producing anomaly scores and rankings rather than binary labels.

---

## Dataset
- **CICIDS2017** (Canadian Institute for Cybersecurity, University of New Brunswick)
- Pre-extracted flow-level features in CSV format (78 features per flow)
- ~2.8 million flow records across 8 CSV files
- Labels: BENIGN + 14 attack types (DoS, DDoS, Brute Force, Web Attacks, Botnet, etc.)
- Citation: Sharafaldin, Lashkari and Ghorbani (2018)
- Download: https://www.unb.ca/cic/datasets/ids-2017.html

---

## Mathematical Framework (Chapter 2)

### Problem Definition
- Dataset: X = {x₁, x₂, ..., xₙ}, xᵢ ∈ ℝᵈ
- Each xᵢ is a d-dimensional flow-level feature vector
- Goal: learn scoring function s: ℝᵈ → ℝ (higher score = more anomalous)
- No labels used during training (unsupervised)

### Key Assumptions
1. Anomalies are RARE (minority class)
2. Anomalies occupy structurally DISTINCT regions of feature space

---

## Three Models to Implement

### 1. Isolation Forest (sklearn)
- Anomaly score: s(xᵢ, n) = 2^(-h̄(xᵢ)/c(n))
- Key parameter: n_estimators=100 (number of trees t)
- Import: from sklearn.ensemble import IsolationForest

### 2. Local Outlier Factor (sklearn)
- Anomaly score: LOF_k(xᵢ) = mean ratio of neighbour density to own density
- Key parameter: n_neighbors=20 (k)
- Import: from sklearn.neighbors import LocalOutlierFactor

### 3. Autoencoder (TensorFlow/Keras)
- Anomaly score: s(xᵢ) = ‖xᵢ - x̂ᵢ‖² (reconstruction error)
- Architecture: d → d/2 → d/4 → m → d/4 → d/2 → d
- Activation: ReLU (hidden), Linear (output)
- Loss: MSE, Optimiser: Adam
- Train on BENIGN traffic only

---

## Evaluation Metrics

| Metric | Purpose | Function |
|--------|---------|----------|
| AUC-ROC | Primary metric, threshold-independent | roc_auc_score() |
| Average Precision | Handles class imbalance | average_precision_score() |
| Precision@k | Operational utility (k=50,100,200) | Custom function |
| Score Distribution | Qualitative analysis | Histogram/ECDF plots |

---

## Project File Structure
```
anomaly_detection/
├── data/
│   └── CICIDS2017/          ← Put CSV files here
├── src/
│   ├── preprocess.py        ← Data loading and preprocessing
│   ├── features.py          ← Feature engineering
│   ├── models/
│   │   ├── isolation_forest.py
│   │   ├── lof.py
│   │   └── autoencoder.py
│   ├── evaluate.py          ← Evaluation metrics
│   └── visualize.py         ← Visualisation
├── notebooks/
│   └── main.ipynb           ← Main notebook
└── requirements.txt
```

---

## Key Features to Extract/Use
- Flow duration
- Packet length (mean, std, max, min) — forward and backward
- Inter-arrival time (mean, std) — forward and backward
- Total bytes/packets — forward and backward
- Upload/download ratio
- TCP flag counts

---

## Important Notes
1. This is UNSUPERVISED — models train on BENIGN traffic only
2. Labels are used ONLY for evaluation (AUC-ROC etc.)
3. Concept drift is treated as a LIMITATION, not an implementation task
4. Output is anomaly SCORES and RANKINGS, not binary labels
5. CICIDS2017 has known labelling issues — acknowledge in discussion

---

## Dependencies
```
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
tensorflow==2.13.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
```

---

## Academic References
- Liu, Ting and Zhou (2008) — Isolation Forest
- Breunig et al. (2000) — LOF
- Sharafaldin et al. (2018) — CICIDS2017 dataset
- Chandola, Banerjee and Kumar (2009) — Anomaly detection survey
