import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import IsolationForest
from evaluate import evaluate_model


class IsolationForestDetector:

    def __init__(self, n_estimators=100, contamination=0.05, random_state=42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit(self, X_train):
        print(f"Training Isolation Forest with {self.n_estimators} trees...")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train)
        print("Training complete.")

    def predict_scores(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        scores = -self.model.score_samples(X)
        return scores

    def evaluate(self, X_test, y_test, k_values=[50, 100, 200]):
        print("Scoring test set with Isolation Forest...")
        scores = self.predict_scores(X_test)
        results = evaluate_model(
            model_name="Isolation Forest",
            y_true=y_test,
            scores=scores,
            k_values=k_values
        )
        return scores, results


def run_isolation_forest(X_train, X_test, y_test,
                         n_estimators=100,
                         contamination=0.05,
                         k_values=[50, 100, 200]):
    detector = IsolationForestDetector(
        n_estimators=n_estimators,
        contamination=contamination
    )
    detector.fit(X_train)
    scores, results = detector.evaluate(X_test, y_test, k_values)
    return scores, results, detector