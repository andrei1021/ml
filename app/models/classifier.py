from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

from app.config import DEFAULT_MODEL_PATH
from app.features.extractor import NUMERIC_FEATURES


@dataclass
class SimpleSoftmaxClassifier:
    """Lightweight softmax regression for environments without scikit-learn."""

    learning_rate: float = 0.01
    epochs: int = 400
    l2: float = 0.001
    weights: np.ndarray | None = field(default=None, init=False)
    bias: np.ndarray | None = field(default=None, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    std_: np.ndarray | None = field(default=None, init=False)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Model is not fitted yet")
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleSoftmaxClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xn = self._standardize(X)

        n_samples, n_features = Xn.shape
        n_classes = int(y.max()) + 1

        rng = np.random.default_rng(42)
        self.weights = rng.normal(scale=0.01, size=(n_features, n_classes))
        self.bias = np.zeros(n_classes)

        for _ in range(self.epochs):
            logits = Xn @ self.weights + self.bias
            probs = self._softmax(logits)

            targets = np.zeros_like(probs)
            targets[np.arange(n_samples), y] = 1.0

            error = probs - targets
            grad_w = (Xn.T @ error) / n_samples + self.l2 * self.weights
            grad_b = error.mean(axis=0)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if any(attr is None for attr in (self.weights, self.bias, self.mean_, self.std_)):
            raise ValueError("Model is not fitted yet")

        X = np.asarray(X, dtype=float)
        Xn = self._standardize(X)
        logits = Xn @ self.weights + self.bias
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


class MatchOutcomeModel:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model: SimpleSoftmaxClassifier | None = None
        self.feature_count = len(NUMERIC_FEATURES) + 5

    def create_default_model(self, feature_count: int | None = None) -> SimpleSoftmaxClassifier:
        self.feature_count = feature_count or self.feature_count
        return SimpleSoftmaxClassifier()

    def load_or_create(self, feature_count: int | None = None) -> SimpleSoftmaxClassifier:
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            self.model = self.create_default_model(feature_count)
        return self.model

    def save(self) -> None:
        if self.model is None:
            raise ValueError("Model not trained")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def predict_proba(self, features: np.ndarray) -> Tuple[float, float, float]:
        if self.model is None:
            self.load_or_create(feature_count=features.shape[1])
        assert self.model is not None
        proba = self.model.predict_proba(features)[0]

        if proba.shape[0] == 2:
            home_win = float(proba[1])
            away_win = float(proba[0] * 0.3)
            draw = float(max(0.0, 1.0 - home_win - away_win))
            return home_win, away_win, draw

        home_win = float(proba[0])
        away_win = float(proba[1])
        draw = float(proba[2]) if proba.shape[0] > 2 else 0.0
        return home_win, away_win, draw
