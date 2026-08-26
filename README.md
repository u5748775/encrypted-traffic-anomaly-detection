# Anomaly Detection in Encrypted Network Traffic

MSc dissertation project (University of Warwick) investigating unsupervised machine learning approaches for detecting anomalous behaviour in encrypted network traffic **without payload inspection**, using only flow-level statistical features.

Three unsupervised models are implemented and compared:

- **Isolation Forest** (`sklearn.ensemble.IsolationForest`)
- **Local Outlier Factor (LOF)** (`sklearn.neighbors.LocalOutlierFactor`)
- **Autoencoder** (TensorFlow/Keras, trained on benign traffic only)

Each model outputs continuous anomaly scores/rankings rather than binary labels; ground-truth labels are used only for evaluation.

## Project Structure

```
anomaly_detection/
├── data/
│   └── CICIDS2017/          # Raw dataset CSVs (not included, see below)
├── src/
│   ├── preprocess.py         # Data loading and preprocessing
│   ├── evaluate.py           # Evaluation metrics
│   ├── visualize.py          # Plots and visualisations
│   └── models/
│       ├── isolation_forest.py
│       ├── lof.py
│       └── autoencoder.py
├── notebooks/
│   └── main.ipynb            # Main pipeline notebook
├── outputs/
│   ├── results.csv           # Model comparison metrics
│   └── figures/               # Generated plots
├── requirements.txt
└── PROJECT_CONTEXT.md         # Detailed methodology/mathematical framework
```

## Dataset

This project uses **CICIDS2017** (Canadian Institute for Cybersecurity, University of New Brunswick):

- Pre-extracted flow-level features in CSV format (78 features per flow)
- ~2.8 million flow records across 8 CSV files
- Labels: BENIGN + 14 attack types (DoS, DDoS, Brute Force, Web Attacks, Botnet, etc.)
- Download: https://www.unb.ca/cic/datasets/ids-2017.html
- Citation: Sharafaldin, Lashkari and Ghorbani (2018)

> **Note:** The raw dataset is not included in this repository due to its size. Download the CSV files from the link above and place them under `data/CICIDS2017/` before running the notebook.

## Setup

```bash
pip install -r requirements.txt
```

## Running

The full pipeline (preprocessing, training, evaluation, and visualisation) is run from the main notebook:

```bash
jupyter notebook notebooks/main.ipynb
```

## Results Summary

Evaluation results from `outputs/results.csv`:

| Model | AUC-ROC | Average Precision | Precision@50 | Precision@100 | Precision@200 | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.8057 | 0.8145 | 0.20 | 0.10 | 0.055 | 0.9148 | 0.0909 | 0.1654 |
| Local Outlier Factor | 0.4321 | 0.4708 | 0.50 | 0.61 | 0.535 | 0.4980 | 0.0492 | 0.0896 |
| Autoencoder | 0.7755 | 0.8282 | 0.34 | 0.38 | 0.655 | 0.9825 | 0.0975 | 0.1774 |

Isolation Forest and the Autoencoder substantially outperform LOF on this dataset in terms of AUC-ROC and Average Precision, while LOF performs close to random. See `outputs/figures/` for ROC/PR curves and score distribution plots, and `PROJECT_CONTEXT.md` for the full methodology.
