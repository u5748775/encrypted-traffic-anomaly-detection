import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Set consistent plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'Isolation Forest': '#2E86AB',
    'Local Outlier Factor': '#A23B72',
    'Autoencoder': '#F18F01'
}


# ── 1. Anomaly Score Distribution ────────────────────────────────
def plot_score_distribution(scores_dict, y_test, save_path=None):
    """
    Plot the anomaly score distribution for each model.

    Corresponds to Section 2.6.5 in Chapter 2:
    A well-performing model should produce a bimodal distribution,
    with normal flows scoring low and anomalous flows scoring high.

    Parameters:
        scores_dict: dict of {model_name: scores array}
        y_test: ground truth labels (0 = normal, 1 = anomalous)
        save_path: file path to save the figure (optional)
    """
    n_models = len(scores_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, scores) in zip(axes, scores_dict.items()):
        color = COLORS.get(model_name, '#333333')

        # Normalise scores to [0, 1] for comparability.
        # Clip to the 99th percentile before normalising so that a small
        # number of extreme outlier scores (common with LOF) do not
        # compress the rest of the distribution against zero.
        p99 = np.percentile(scores, 99)
        scores_clipped = np.clip(scores, scores.min(), p99)
        scores_norm = (scores_clipped - scores_clipped.min()) / (scores_clipped.max() - scores_clipped.min() + 1e-10)

        # Plot normal and anomalous distributions separately
        ax.hist(scores_norm[y_test == 0], bins=80, alpha=0.6,
                color='steelblue', label='Normal', density=True)
        ax.hist(scores_norm[y_test == 1], bins=80, alpha=0.6,
                color='crimson', label='Anomalous', density=True)

        ax.set_title(f'{model_name}\nAnomaly Score Distribution',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalised Anomaly Score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ── 2. ROC Curves ─────────────────────────────────────────────────
def plot_roc_curves(scores_dict, y_test, save_path=None):
    """
    Plot ROC curves for all models on a single figure.

    Corresponds to Section 2.6.2 in Chapter 2.
    Each curve shows the TPR vs FPR trade-off across all thresholds.
    AUC-ROC is shown in the legend for each model.

    Parameters:
        scores_dict: dict of {model_name: scores array}
        y_test: ground truth labels
        save_path: file path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, scores in scores_dict.items():
        color = COLORS.get(model_name, '#333333')
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_score = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{model_name} (AUC = {auc_score:.4f})')

    # Plot random classifier baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ── 3. Precision-Recall Curves ────────────────────────────────────
def plot_pr_curves(scores_dict, y_test, save_path=None):
    """
    Plot Precision-Recall curves for all models.

    Corresponds to Section 2.6.3 in Chapter 2.
    More informative than ROC under class imbalance.

    Parameters:
        scores_dict: dict of {model_name: scores array}
        y_test: ground truth labels
        save_path: file path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, scores in scores_dict.items():
        color = COLORS.get(model_name, '#333333')
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = auc(recall, precision)

        ax.plot(recall, precision, color=color, linewidth=2,
                label=f'{model_name} (AP = {ap:.4f})')

    # Baseline: random classifier
    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'Random (AP = {baseline:.4f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — Model Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ── 4. Precision@k Bar Chart ──────────────────────────────────────
def plot_precision_at_k(all_results, k_values=[50, 100, 200], save_path=None):
    """
    Plot Precision@k comparison across models as a grouped bar chart.

    Corresponds to Section 2.6.4 in Chapter 2.
    Shows how many true anomalies appear in the top-k flagged flows.

    Parameters:
        all_results: list of result dicts from evaluate_model()
        k_values: list of k values to plot
        save_path: file path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    n_models = len(all_results)
    n_k = len(k_values)
    bar_width = 0.25
    x = np.arange(n_k)

    for i, result in enumerate(all_results):
        model_name = result['model']
        color = COLORS.get(model_name, '#333333')
        values = [result.get(f'precision@{k}', 0) for k in k_values]
        offset = (i - n_models / 2 + 0.5) * bar_width

        bars = ax.bar(x + offset, values, bar_width,
                      label=model_name, color=color, alpha=0.85)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('k (number of top-ranked flows inspected)', fontsize=12)
    ax.set_ylabel('Precision@k', fontsize=12)
    ax.set_title('Precision@k — Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ── 5. Autoencoder Training Curve ────────────────────────────────
def plot_training_curve(history, save_path=None):
    """
    Plot the autoencoder training and validation loss curves.

    Used to verify that the model converged during training
    and did not overfit to the normal training data.

    Parameters:
        history: dict with 'loss' and 'val_loss' lists
                 from AutoencoderDetector.get_training_history()
        save_path: file path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history['loss']) + 1)

    ax.plot(epochs, history['loss'], color='#2E86AB',
            linewidth=2, label='Training Loss')
    ax.plot(epochs, history['val_loss'], color='#F18F01',
            linewidth=2, linestyle='--', label='Validation Loss')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Autoencoder Training Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ── 6. Summary Dashboard ──────────────────────────────────────────
def plot_summary_dashboard(scores_dict, y_test, all_results, save_path=None):
    """
    Plot a combined summary dashboard with all key visualisations:
        - ROC curves
        - PR curves
        - Precision@k bar chart
        - Score distributions

    Provides a single comprehensive figure for the dissertation.

    Parameters:
        scores_dict: dict of {model_name: scores array}
        y_test: ground truth labels
        all_results: list of result dicts from evaluate_model()
        save_path: file path to save the figure (optional)
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── ROC Curves ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for model_name, scores in scores_dict.items():
        color = COLORS.get(model_name, '#333333')
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_score = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{model_name[:2]}... (AUC={auc_score:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_title('ROC Curves', fontweight='bold')
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.legend(fontsize=8)

    # ── PR Curves ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for model_name, scores in scores_dict.items():
        color = COLORS.get(model_name, '#333333')
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = auc(recall, precision)
        ax2.plot(recall, precision, color=color, linewidth=2,
                 label=f'{model_name[:2]}... (AP={ap:.3f})')
    ax2.set_title('Precision-Recall Curves', fontweight='bold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(fontsize=8)

    # ── Precision@k ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    k_values = [50, 100, 200]
    n_models = len(all_results)
    bar_width = 0.25
    x = np.arange(len(k_values))
    for i, result in enumerate(all_results):
        color = COLORS.get(result['model'], '#333333')
        values = [result.get(f'precision@{k}', 0) for k in k_values]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax3.bar(x + offset, values, bar_width,
                label=result['model'][:12], color=color, alpha=0.85)
    ax3.set_title('Precision@k', fontweight='bold')
    ax3.set_xlabel('k')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'k={k}' for k in k_values])
    ax3.set_ylim([0, 1.1])
    ax3.legend(fontsize=8)

    # ── Score Distributions ───────────────────────────────────────
    for idx, (model_name, scores) in enumerate(scores_dict.items()):
        ax = fig.add_subplot(gs[1, idx])
        color = COLORS.get(model_name, '#333333')
        # Same 99th-percentile clipping as plot_score_distribution, to
        # keep the dashboard's score panels consistent and readable.
        p99 = np.percentile(scores, 99)
        scores_clipped = np.clip(scores, scores.min(), p99)
        scores_norm = (scores_clipped - scores_clipped.min()) / (
            scores_clipped.max() - scores_clipped.min() + 1e-10)
        ax.hist(scores_norm[y_test == 0], bins=60, alpha=0.6,
                color='steelblue', label='Normal', density=True)
        ax.hist(scores_norm[y_test == 1], bins=60, alpha=0.6,
                color='crimson', label='Anomalous', density=True)
        ax.set_title(f'{model_name}\nScore Distribution', fontweight='bold')
        ax.set_xlabel('Normalised Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    fig.suptitle(
        'Anomaly Detection in Encrypted Network Traffic — Model Comparison',
        fontsize=14, fontweight='bold', y=1.01
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")

    plt.show()