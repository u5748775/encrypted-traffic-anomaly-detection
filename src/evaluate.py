import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve
)


# ── 1. AUC-ROC ────────────────────────────────────────────────────
def compute_auc_roc(y_true, scores):
    """
    Compute the Area Under the ROC Curve (AUC-ROC).

    This is the primary evaluation metric in this project.
    Corresponds to P(s(x+) > s(x-)) as defined in Chapter 2, Section 2.6.2.

    Parameters:
        y_true: ground truth binary labels (0 = normal, 1 = anomalous)
        scores: anomaly scores produced by the model (higher = more anomalous)

    Returns:
        auc: AUC-ROC score in [0, 1]
    """
    return roc_auc_score(y_true, scores)


# ── 2. Average Precision ──────────────────────────────────────────
def compute_average_precision(y_true, scores):
    """
    Compute Average Precision (AP) from the Precision-Recall curve.

    More informative than AUC-ROC under class imbalance, as it focuses
    exclusively on the anomalous class.
    Corresponds to Section 2.6.3 in Chapter 2.

    Parameters:
        y_true: ground truth binary labels
        scores: anomaly scores

    Returns:
        ap: Average Precision score in [0, 1]
    """
    return average_precision_score(y_true, scores)


# ── 3. Precision at k ─────────────────────────────────────────────
def compute_precision_at_k(y_true, scores, k):
    """
    Compute Precision at k (P@k).

    Evaluates the proportion of true anomalies among the top-k
    highest-scoring observations. Reflects operational triage utility.
    Corresponds to Section 2.6.4 in Chapter 2.

    P@k = |{top-k flagged} ∩ {true anomalies}| / k

    Parameters:
        y_true: ground truth binary labels
        scores: anomaly scores
        k: number of top-ranked observations to evaluate

    Returns:
        precision_at_k: proportion of true anomalies in the top-k
    """
    # Get indices of top-k highest scores
    top_k_indices = np.argsort(scores)[::-1][:k]

    # Count true anomalies in top-k
    true_positives_at_k = y_true[top_k_indices].sum()

    return true_positives_at_k / k


# ── 4. Threshold-dependent metrics ───────────────────────────────
def compute_threshold_metrics(y_true, scores, threshold=None, percentile=95):
    """
    Compute Precision, Recall, and F1 at a given threshold.

    If no threshold is provided, uses the given percentile of the
    score distribution as the threshold (e.g., top 5% flagged as anomalous).

    Parameters:
        y_true: ground truth binary labels
        scores: anomaly scores
        threshold: decision threshold (optional)
        percentile: percentile of scores to use as threshold if none provided

    Returns:
        dict containing precision, recall, f1, and threshold used
    """
    if threshold is None:
        threshold = np.percentile(scores, percentile)

    # Apply threshold to produce binary predictions
    y_pred = (scores >= threshold).astype(int)

    return {
        'threshold': threshold,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


# ── 5. Full evaluation report ─────────────────────────────────────
def evaluate_model(model_name, y_true, scores, k_values=[50, 100, 200]):
    """
    Run the full evaluation pipeline for a single model.

    Computes all metrics defined in Chapter 2, Section 2.6:
        - AUC-ROC (primary metric)
        - Average Precision
        - Precision@k for k in {50, 100, 200}
        - Threshold-dependent metrics (Precision, Recall, F1)

    Parameters:
        model_name: name of the model (for display)
        y_true: ground truth binary labels
        scores: anomaly scores
        k_values: list of k values for Precision@k

    Returns:
        results: dictionary of all computed metrics
    """
    results = {
        'model': model_name,
        'auc_roc': compute_auc_roc(y_true, scores),
        'average_precision': compute_average_precision(y_true, scores),
    }

    # Precision@k for each k
    for k in k_values:
        if k <= len(scores):
            results[f'precision@{k}'] = compute_precision_at_k(y_true, scores, k)

    # Threshold-dependent metrics at 95th percentile
    threshold_metrics = compute_threshold_metrics(y_true, scores)
    results.update(threshold_metrics)

    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*50}")
    print(f"AUC-ROC:            {results['auc_roc']:.4f}")
    print(f"Average Precision:  {results['average_precision']:.4f}")
    for k in k_values:
        if f'precision@{k}' in results:
            print(f"Precision@{k}:      {results[f'precision@{k}']:.4f}")
    print(f"Threshold:          {results['threshold']:.4f}")
    print(f"Precision:          {results['precision']:.4f}")
    print(f"Recall:             {results['recall']:.4f}")
    print(f"F1-Score:           {results['f1']:.4f}")

    return results


# ── 6. Compare all models ─────────────────────────────────────────
def compare_models(all_results):
    """
    Print a summary comparison table of all models.

    Parameters:
        all_results: list of result dictionaries from evaluate_model()
    """
    print(f"\n{'='*70}")
    print("Model Comparison Summary")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'AUC-ROC':>10} {'Avg Prec':>10} "
          f"{'P@50':>8} {'P@100':>8} {'P@200':>8} {'F1':>8}")
    print(f"{'-'*70}")

    for r in all_results:
        print(
            f"{r['model']:<20} "
            f"{r['auc_roc']:>10.4f} "
            f"{r['average_precision']:>10.4f} "
            f"{r.get('precision@50', 0):>8.4f} "
            f"{r.get('precision@100', 0):>8.4f} "
            f"{r.get('precision@200', 0):>8.4f} "
            f"{r['f1']:>8.4f}"
        )
    print(f"{'='*70}")