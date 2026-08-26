import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import evaluate_model

import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# ── Autoencoder Model ─────────────────────────────────────────────
class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.

    Implements the algorithm described in Chapter 2, Section 2.5.

    Architecture (symmetric encoder-decoder):
        d → d/2 → d/4 → m → d/4 → d/2 → d

    Training objective:
        ℒ(θ, ϕ) = (1/n) Σ ‖xᵢ - g_ϕ(f_θ(xᵢ))‖²

    Anomaly score:
        s(xᵢ) = ‖xᵢ - x̂ᵢ‖² (reconstruction error)

    Key insight: trained on normal traffic only, the model learns
    to reconstruct normal patterns accurately. Anomalous flows
    produce high reconstruction errors.

    Reference: Zong et al. (2018), Ruff et al. (2018)
    """

    def __init__(self, input_dim, encoding_dim=8,
                 learning_rate=0.001, epochs=50, batch_size=256):
        """
        Initialise the Autoencoder detector.

        Parameters:
            input_dim: number of input features (d in Chapter 2)
                       Must match the number of features in X_train
            encoding_dim: dimension of the bottleneck layer (m in Chapter 2)
                          Controls how compressed the latent representation is
                          Smaller = more compression, but may lose information
            learning_rate: learning rate η for Adam optimiser
                           Corresponds to η in the gradient update in Chapter 2
            epochs: maximum number of training epochs
            batch_size: number of samples per gradient update (mini-batch SGD)
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None

    def build_model(self):
        """
        Build the autoencoder architecture.

        Implements the symmetric encoder-decoder structure from Chapter 2:
            d → d/2 → d/4 → m → d/4 → d/2 → d

        Encoder: compresses input to latent representation z
            f_θ: ℝᵈ → ℝᵐ

        Decoder: reconstructs input from latent representation
            g_ϕ: ℝᵐ → ℝᵈ

        Activation: ReLU for hidden layers (σ(x) = max(0, x))
                    Linear for output layer
        """
    def build_model(self):
        """
        Build the autoencoder architecture.

        Implements the symmetric encoder-decoder structure from Chapter 2:
            d → d/2 → d/4 → m → d/4 → d/2 → d

        Encoder: compresses input to latent representation z
            f_θ: ℝᵈ → ℝᵐ

        Decoder: reconstructs input from latent representation
            g_ϕ: ℝᵐ → ℝᵈ

        Activation: ReLU for hidden layers (σ(x) = max(0, x))
                    Linear for output layer
        """
        # Fix random seeds so weight initialisation is reproducible
        # across runs (see Section 3.10.3 / 6.5)
        tf.random.set_seed(42)
        np.random.seed(42)
        d = self.input_dim
        m = self.encoding_dim

        # ── Input layer ───────────────────────────────────────────
        inputs = Input(shape=(d,), name='input')

        # ── Encoder: d → d/2 → d/4 → m ──────────────────────────
        # Each layer applies: h⁽ˡ⁾ = σ(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)
        x = Dense(max(d // 2, m * 4), activation='relu',
                  name='encoder_1')(inputs)
        x = Dense(max(d // 4, m * 2), activation='relu',
                  name='encoder_2')(x)

        # ── Bottleneck layer: latent representation z ─────────────
        # This is zᵢ = f_θ(xᵢ) in Chapter 2
        encoded = Dense(m, activation='relu',
                        name='bottleneck')(x)

        # ── Decoder: m → d/4 → d/2 → d ──────────────────────────
        x = Dense(max(d // 4, m * 2), activation='relu',
                  name='decoder_1')(encoded)
        x = Dense(max(d // 2, m * 4), activation='relu',
                  name='decoder_2')(x)

        # ── Output layer: reconstruction x̂ᵢ = g_ϕ(zᵢ) ──────────
        # Linear activation to allow any real-valued reconstruction
        outputs = Dense(d, activation='linear',
                        name='output')(x)

        # ── Build full autoencoder model ──────────────────────────
        self.model = Model(inputs=inputs, outputs=outputs,
                           name='autoencoder')

        # ── Compile with Adam optimiser and MSE loss ──────────────
        # Corresponds to: ℒ(θ, ϕ) = (1/n) Σ ‖xᵢ - x̂ᵢ‖²
        # Adam implements the adaptive gradient update from Chapter 2
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        # Print model summary
        self.model.summary()

    def fit(self, X_train, validation_split=0.1):
        """
        Train the autoencoder on normal traffic only.

        Uses early stopping to prevent overfitting:
        training stops when validation loss stops improving.

        Parameters:
            X_train: training feature matrix (normal traffic only)
                     shape (n, d) — corresponds to unlabelled 𝒟 in Chapter 2
            validation_split: proportion of training data used for validation
        """
        if self.model is None:
            self.build_model()

        print(f"\nTraining Autoencoder for up to {self.epochs} epochs...")
        print(f"Input dimension: {self.input_dim}")
        print(f"Bottleneck dimension: {self.encoding_dim}")

        # ── Callbacks ─────────────────────────────────────────────
        callbacks = [
            # Stop training if validation loss does not improve for 10 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]

        # ── Train the model ───────────────────────────────────────
        # Note: input = output (autoencoder reconstructs its own input)
        # This implements: minimise ℒ(θ, ϕ) over training set
        self.history = self.model.fit(
            X_train, X_train,          # input and target are the same
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        final_loss = self.history.history['val_loss'][-1]
        print(f"\nTraining complete. Final validation loss: {final_loss:.6f}")

    def predict_scores(self, X):
        """
        Compute reconstruction error as anomaly score.

        Implements s(xᵢ) = ‖xᵢ - x̂ᵢ‖² from Chapter 2, Section 2.5.3.

        Normal observations: low reconstruction error → low anomaly score
        Anomalous observations: high reconstruction error → high anomaly score

        Parameters:
            X: feature matrix to score, shape (n, d)

        Returns:
            scores: per-sample reconstruction errors, shape (n,)
                    Higher score = more anomalous
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Get reconstructed output x̂ᵢ = g_ϕ(f_θ(xᵢ))
        X_reconstructed = self.model.predict(X, verbose=0)

        # Compute per-sample MSE: s(xᵢ) = ‖xᵢ - x̂ᵢ‖²
        scores = np.mean((X - X_reconstructed) ** 2, axis=1)

        return scores

    def evaluate(self, X_test, y_test, k_values=[50, 100, 200]):
        """
        Compute anomaly scores on the test set and evaluate performance.

        Parameters:
            X_test: test feature matrix (normal + anomalous), shape (n, d)
            y_test: ground truth labels (0 = normal, 1 = anomalous)
            k_values: list of k values for Precision@k

        Returns:
            scores: anomaly scores for all test observations
            results: dictionary of evaluation metrics
        """
        print("\nScoring test set with Autoencoder...")
        scores = self.predict_scores(X_test)

        results = evaluate_model(
            model_name="Autoencoder",
            y_true=y_test,
            scores=scores,
            k_values=k_values
        )

        return scores, results

    def get_training_history(self):
        """
        Return training and validation loss history.

        Useful for plotting the learning curve to verify
        that the model converged properly during training.

        Returns:
            history: dict with 'loss' and 'val_loss' lists
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet.")

        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss']
        }


# ── Convenience function ──────────────────────────────────────────
def run_autoencoder(X_train, X_test, y_test,
                    encoding_dim=8,
                    learning_rate=0.001,
                    epochs=50,
                    batch_size=256,
                    k_values=[50, 100, 200]):
    """
    End-to-end Autoencoder pipeline: build → train → score → evaluate.

    Parameters:
        X_train: training feature matrix (normal traffic only)
        X_test: test feature matrix (normal + anomalous)
        y_test: ground truth labels
        encoding_dim: bottleneck dimension m
        learning_rate: Adam learning rate η
        epochs: maximum training epochs
        batch_size: mini-batch size
        k_values: list of k values for Precision@k

    Returns:
        scores: anomaly scores for test set
        results: dictionary of evaluation metrics
        detector: trained AutoencoderDetector object
    """
    input_dim = X_train.shape[1]

    detector = AutoencoderDetector(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

    detector.fit(X_train)
    scores, results = detector.evaluate(X_test, y_test, k_values)

    return scores, results, detector