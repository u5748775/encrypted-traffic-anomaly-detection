# Anomaly Detection in Encrypted Network Traffic

MSc dissertation project (University of Warwick) investigating unsupervised machine learning approaches for detecting anomalous behaviour in encrypted network traffic **without payload inspection**, using only flow-level statistical features.

## Background

Deep packet inspection is increasingly infeasible against encrypted traffic, so this project asks whether anomalies can still be detected purely from **flow-level metadata** (packet lengths, timing, byte/packet counts, etc.) rather than payload content. The problem is framed as **unsupervised anomaly scoring**: given a dataset of flow feature vectors X = {x₁, ..., xₙ}, learn a scoring function s: ℝᵈ → ℝ (higher = more anomalous) without using labels during training. Labels are used only at evaluation time.

Two working assumptions underpin the approach:
1. Anomalies are **rare** relative to normal traffic.
2. Anomalies occupy **structurally distinct** regions of the feature space.

## Models

Three unsupervised models are implemented and compared, each with its own module under [src/models/](src/models/):

| Model | Approach | Key parameters | Reference |
|---|---|---|---|
| [Isolation Forest](src/models/isolation_forest.py) | Global isolation-based; anomalies are easier to isolate via random partitioning | `n_estimators=100`, `contamination=0.05` | Liu, Ting & Zhou (2008) |
| [Local Outlier Factor (LOF)](src/models/lof.py) | Local density-based; compares a point's density to its neighbours' | `n_neighbors=20`, `contamination=0.05`, transductive (fit and score on the same data) | Breunig et al. (2000) |
| [Autoencoder](src/models/autoencoder.py) | Reconstruction-based; trained on **benign traffic only**, anomaly score = reconstruction error | Symmetric encoder-decoder `d → d/2 → d/4 → m → d/4 → d/2 → d` (`m=8`), ReLU hidden / linear output, Adam (`lr=0.001`), MSE loss, early stopping on validation loss | Zong et al. (2018), Ruff et al. (2018) |

All three output continuous anomaly scores/rankings rather than binary labels.

## Evaluation Metrics

Implemented in [src/evaluate.py](src/evaluate.py):

- **AUC-ROC** — primary, threshold-independent metric
- **Average Precision** — more informative than AUC-ROC under class imbalance
- **Precision@k** (k = 50, 100, 200) — proportion of true anomalies among the top-k highest-scoring flows; reflects operational triage utility
- **Precision / Recall / F1** — computed at a fixed decision threshold (95th percentile of the score distribution)

## Project Structure

```
anomaly_detection/
├── data/
│   └── CICIDS2017/          # Raw dataset CSVs (not included, see below)
├── src/
│   ├── preprocess.py         # Data loading, cleaning, scaling, train/test split
│   ├── evaluate.py           # Evaluation metrics (AUC-ROC, AP, Precision@k, etc.)
│   ├── visualize.py          # ROC/PR curves, score distributions, dashboards
│   └── models/
│       ├── isolation_forest.py
│       ├── lof.py
│       └── autoencoder.py
├── notebooks/
│   └── main.ipynb            # Main pipeline notebook: run everything end-to-end
├── outputs/
│   ├── results.csv           # Model comparison metrics
│   ├── feature_importance.csv / feature_statistics.csv / attack_distribution.csv
│   └── figures/               # Generated plots (ROC/PR curves, dashboards, etc.)
├── requirements.txt
└── PROJECT_CONTEXT.md         # Full methodology and mathematical framework
```

## Dataset

This project uses **CICIDS2017** (Canadian Institute for Cybersecurity, University of New Brunswick):

- Pre-extracted flow-level features in CSV format (78 features per flow)
- ~2.8 million flow records across 8 CSV files (one per day/attack scenario)
- Labels: BENIGN + 14 attack types (DoS, DDoS, Brute Force, Web Attacks, Botnet, PortScan, Infiltration, etc.)
- Key feature groups: flow duration; forward/backward packet length (mean/std/max/min); forward/backward inter-arrival time; total bytes/packets; upload/download ratio; TCP flag counts
- Download: https://www.unb.ca/cic/datasets/ids-2017.html
- Citation: Sharafaldin, Lashkari and Ghorbani (2018)

> **Note:** The raw dataset is not included in this repository due to its size (~2.8M rows, several GB across 8 CSVs). Download the CSV files from the link above and place them under `data/CICIDS2017/` before running the notebook.

## Setup

```bash
pip install -r requirements.txt
```

## Running

The full pipeline (preprocessing, training, evaluation, and visualisation) is run from the main notebook:

```bash
jupyter notebook notebooks/main.ipynb
```

This loads the CICIDS2017 CSVs from `data/CICIDS2017/`, preprocesses and splits the data (`src/preprocess.py`), trains and scores each model (`src/models/`), computes metrics (`src/evaluate.py`), and generates the figures in `outputs/figures/` (`src/visualize.py`).

## Results Summary

Evaluation results from [outputs/results.csv](outputs/results.csv):

| Model | AUC-ROC | Average Precision | Precision@50 | Precision@100 | Precision@200 | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.8057 | 0.8145 | 0.20 | 0.10 | 0.055 | 0.9148 | 0.0909 | 0.1654 |
| Local Outlier Factor | 0.4321 | 0.4708 | 0.50 | 0.61 | 0.535 | 0.4980 | 0.0492 | 0.0896 |
| Autoencoder | 0.7755 | 0.8282 | 0.34 | 0.38 | 0.655 | 0.9825 | 0.0975 | 0.1774 |

**Interpretation:**
- **Isolation Forest** and the **Autoencoder** substantially outperform LOF in ranking quality (AUC-ROC ≈ 0.78–0.81, Average Precision ≈ 0.81–0.83), and both achieve very high precision (≥ 0.91) at the 95th-percentile threshold — few false positives among flagged flows, but a low recall reflects that only the top ~5% of scores are flagged.
- **LOF** performs close to random on AUC-ROC (0.43), though it still achieves comparatively higher Precision@k at larger k — its local density formulation appears less suited to this dataset's global structure than the other two methods.
- Full ROC/PR curves, score distributions, and a summary dashboard are available in [outputs/figures/](outputs/figures/); the full methodology and mathematical derivations are in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## Limitations

- This is an **unsupervised** setup: labels are used only for evaluation, not training (the Autoencoder is trained on benign traffic only).
- **Concept drift** is discussed as a limitation of the approach rather than something handled by the implementation.
- CICIDS2017 has known label-quality issues documented in the literature, which should be kept in mind when interpreting results.

## References

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers.
- Zong, B. et al. (2018). Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection.
- Ruff, L. et al. (2018). Deep One-Class Classification.
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization (CICIDS2017).
- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey.
