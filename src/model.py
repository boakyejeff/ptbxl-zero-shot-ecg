"""Small MLP embedding model (sklearn, CPU) trained 4-way per LOSO fold."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class EmbeddingMLP:
    """Standardized features -> one hidden layer -> softmax over known classes.

    Penultimate-layer (hidden) activations are the embeddings used for OOD
    scoring and few-shot nearest-centroid classification.
    """

    def __init__(
        self,
        hidden: int = 96,
        alpha: float = 1e-2,
        max_iter: int = 600,
        seed: int = 0,
    ):
        self.hidden = hidden
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(hidden,),
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.15,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmbeddingMLP":
        Xs = self.scaler.fit_transform(X)
        self.mlp.fit(Xs, y)
        self.classes_ = self.mlp.classes_
        return self

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        a = Xs @ self.mlp.coefs_[0] + self.mlp.intercepts_[0]
        return np.maximum(a, 0.0)  # relu

    def embedding(self, X: np.ndarray) -> np.ndarray:
        return self._hidden(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.mlp.predict_proba(self.scaler.transform(X))

    def max_softmax(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).max(axis=1)

    def energy(self, X: np.ndarray) -> np.ndarray:
        """Negative logsumexp of logits; lower = more in-distribution."""
        Xs = self.scaler.transform(X)
        a = self._hidden_raw(Xs)
        logits = a @ self.mlp.coefs_[1] + self.mlp.intercepts_[1]
        m = logits.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
        return -lse

    def _hidden_raw(self, Xs: np.ndarray) -> np.ndarray:
        a = Xs @ self.mlp.coefs_[0] + self.mlp.intercepts_[0]
        return np.maximum(a, 0.0)
