import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from app.config import DEFAULT_MODEL_PATH, RAW_DATA_PATH
from app.features.extractor import NUMERIC_FEATURES
from app.models.classifier import MatchOutcomeModel


def load_training_data(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if "target_outcome" not in df.columns:
        raise ValueError("Dataset requires 'target_outcome' column")

    X = df[NUMERIC_FEATURES + ["context_live", "context_prematch", "context_combo", "sport_football", "sport_basketball"]]
    y = df["target_outcome"]
    return X, y.to_numpy()


def main(dataset: Path, output: Path) -> None:
    model_wrapper = MatchOutcomeModel(output)
    model = model_wrapper.create_default_pipeline()

    df = pd.read_csv(dataset)
    if df.empty:
        raise ValueError("Training dataset is empty")

    encoder = LabelEncoder()
    df["target_label"] = encoder.fit_transform(df["target_outcome"])

    feature_columns = NUMERIC_FEATURES + [
        "context_live",
        "context_prematch",
        "context_combo",
        "sport_football",
        "sport_basketball",
    ]
    X = df[feature_columns]
    y = df["target_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)

    joblib.dump(encoder, output.with_suffix(".labels.joblib"))
    joblib.dump(feature_columns, output.with_suffix(".features.joblib"))

    model_wrapper.pipeline = model
    model_wrapper.save()
    print(f"Model saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train match outcome model")
    parser.add_argument("--dataset", type=Path, default=RAW_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the trained model")
    args = parser.parse_args()
    main(args.dataset, args.output)
