import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.neighbors import LocalOutlierFactor
from evaluate import evaluate_model


# ── Local Outlier Factor Model ────────────────────────────────────
class LOFDetector:
    """
    Local Outlier Factor-based anomaly detector.

    Implements the algorithm described in Chapter 2, Section 2.4.
    Anomaly score LOF_k(xᵢ) compares the local reachability density
    of each observation to that of its k nearest neighbours.

    Key distinction from Isolation Forest:
        - Isolation Forest: global isolation-based detection
        - LOF: local density-based detection
    Both are complementary and used in parallel in this project.

    Reference: Breunig et al. (2000)
    """

    def __init__(self, n_neighbors=20, contamination=0.05):
        """
        Initialise the LOF detector.

        Parameters:
            n_neighbors: number of neighbours k used for density estimation
                         Corresponds to k in the LOF_k formula in Chapter 2.
                         Larger k = smoother density estimates, less sensitive
                         to local noise. Smaller k = more sensitive to fine-
                         grained local structure.
            contamination: expected proportion of anomalies in the dataset
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None

    def fit_predict_scores(self, X):
        """
        Fit the LOF model and compute anomaly scores.

        Note: Unlike Isolation Forest, LOF does not have separate fit()
        and predict() steps in sklearn. It computes scores during fitting,
        which means it uses the entire dataset (train + test together).

        This is a known limitation of LOF: it is a transductive method,
        meaning it cannot score new unseen data without retraining.
        For this project, we apply LOF to the full test set directly.

        Parameters:
            X: feature matrix to fit and score, shape (n, d)

        Returns:
            scores: anomaly scores, shape (n,)
                    Higher score = more anomalous
        """
        print(f"Fitting LOF with k={self.n_neighbors} neighbours...")

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,  # Transductive mode: score training data directly
            n_jobs=-1       # Use all available CPU cores
        )

        # fit_predict returns -1 for anomalies, 1 for normal
        # negative_outlier_factor_ gives the raw LOF scores (negative)
        self.model.fit_predict(X)

        # Negate so that higher scores = more anomalous
        # Consistent with scoring function s: ℝᵈ → ℝ in Chapter 2
        scores = -self.model.negative_outlier_factor_

        print("LOF scoring complete.")
        return scores

    def evaluate(self, X_test, y_test, k_values=[50, 100, 200]):
        """
        Compute LOF anomaly scores on the test set and evaluate performance.

        Parameters:
            X_test: test feature matrix (normal + anomalous), shape (n, d)
            y_test: ground truth labels (0 = normal, 1 = anomalous)
            k_values: list of k values for Precision@k

        Returns:
            scores: anomaly scores for all test observations
            results: dictionary of evaluation metrics
        """
        print("\nScoring test set with LOF...")
        scores = self.fit_predict_scores(X_test)

        results = evaluate_model(
            model_name="Local Outlier Factor",
            y_true=y_test,
            scores=scores,
            k_values=k_values
        )

        return scores, results


# ── Convenience function ──────────────────────────────────────────
def run_lof(X_test, y_test,
            n_neighbors=20,
            contamination=0.05,
            k_values=[50, 100, 200]):
    """
    End-to-end LOF pipeline: fit → score → evaluate.

    Note: LOF is applied directly to the test set (transductive).
    It does not use X_train separately because it computes
    density estimates from the data it is given.

    Parameters:
        X_test: test feature matrix (normal + anomalous)
        y_test: ground truth labels
        n_neighbors: number of neighbours k
        contamination: expected anomaly proportion
        k_values: list of k values for Precision@k

    Returns:
        scores: anomaly scores for test set
        results: dictionary of evaluation metrics
        detector: trained LOFDetector object
    """
    detector = LOFDetector(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    scores, results = detector.evaluate(X_test, y_test, k_values)

    return scores, results, detector