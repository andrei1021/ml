from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import DEFAULT_MODEL_PATH
from app.features.extractor import NUMERIC_FEATURES


class MatchOutcomeModel:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Pipeline | None = None

    def create_default_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, list(range(len(NUMERIC_FEATURES) + 5)))]
        )

        base_model = GradientBoostingClassifier(random_state=42)
        calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)

        return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", calibrated_model)])

    def load_or_create(self) -> Pipeline:
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
        else:
            self.pipeline = self.create_default_pipeline()
        return self.pipeline

    def save(self) -> None:
        if self.pipeline is None:
            raise ValueError("Model not trained")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

    def predict_proba(self, features: np.ndarray) -> Tuple[float, float, float]:
        if self.pipeline is None:
            self.load_or_create()
        assert self.pipeline is not None
        proba = self.pipeline.predict_proba(features)[0]
        if proba.shape[0] == 2:
            home_win = proba[1]
            away_win = proba[0] * 0.3
            draw = 1.0 - home_win - away_win
            return float(home_win), float(away_win), float(draw)
        return float(proba[0]), float(proba[1]), float(proba[2])
