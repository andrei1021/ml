import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from app.config import DEFAULT_MODEL_PATH, RAW_DATA_PATH
from app.features.extractor import NUMERIC_FEATURES
from app.models.classifier import MatchOutcomeModel


def factorize_targets(labels: pd.Series) -> Tuple[np.ndarray, List[str]]:
    unique = list(dict.fromkeys(labels))
    index_map = {label: idx for idx, label in enumerate(unique)}
    encoded = labels.map(index_map).to_numpy(dtype=int)
    return encoded, unique


def train_test_split_numpy(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main(dataset: Path, output: Path) -> None:
    model_wrapper = MatchOutcomeModel(output)

    df = pd.read_csv(dataset)
    if df.empty:
        raise ValueError("Training dataset is empty")

    feature_columns = NUMERIC_FEATURES + [
        "context_live",
        "context_prematch",
        "context_combo",
        "sport_football",
        "sport_basketball",
    ]

    missing = [col for col in feature_columns + ["target_outcome"] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df[feature_columns].to_numpy(dtype=float)
    y_encoded, label_classes = factorize_targets(df["target_outcome"])

    X_train, X_test, y_train, y_test = train_test_split_numpy(X, y_encoded, test_size=0.2, seed=42)

    model = model_wrapper.create_default_model(feature_count=X.shape[1])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float((y_pred == y_test).mean()) if len(y_test) > 0 else 0.0
    print(f"Validation accuracy: {accuracy:.3f}")

    joblib.dump(label_classes, output.with_suffix(".labels.joblib"))
    joblib.dump(feature_columns, output.with_suffix(".features.joblib"))

    model_wrapper.model = model
    model_wrapper.save()
    print(f"Model saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train match outcome model")
    parser.add_argument("--dataset", type=Path, default=RAW_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the trained model")
    args = parser.parse_args()
    main(args.dataset, args.output)
