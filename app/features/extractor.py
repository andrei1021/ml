from typing import Any, Dict, List

import numpy as np

from app.data.schema import Odds, PredictionRequest


NUMERIC_FEATURES = [
    "minute",
    "score_home",
    "score_away",
    "xg_home",
    "xg_away",
    "shots_on_target_home",
    "shots_on_target_away",
    "shots_home",
    "shots_away",
    "corners_home",
    "corners_away",
    "dangerous_attacks_home",
    "dangerous_attacks_away",
    "possession_home",
    "possession_away",
    "red_cards_home",
    "red_cards_away",
    "over_1_5_goals",
    "over_2_5_goals",
    "under_2_5_goals",
    "btts_yes",
    "btts_no",
    "over_9_5_corners",
    "under_9_5_corners",
]


def _odds_to_dict(odds: Odds | None) -> Dict[str, float | None]:
    if odds is None:
        return {k: None for k in NUMERIC_FEATURES if "over" in k or "under" in k or "btts" in k}
    return odds.model_dump()


def _safe_get(container: Dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = container.get(key)
    if value is None:
        return default
    return float(value)


def build_feature_vector(request: PredictionRequest) -> List[float]:
    stats = request.stats.model_dump() if request.stats else {}
    score = request.score.model_dump() if request.score else {}
    odds = _odds_to_dict(request.odds)

    values: Dict[str, float | None] = {
        "minute": _safe_get(request.model_dump(), "minute", 0),
        "score_home": _safe_get(score, "home", 0),
        "score_away": _safe_get(score, "away", 0),
        "xg_home": _safe_get(stats, "xg_home", 0),
        "xg_away": _safe_get(stats, "xg_away", 0),
        "shots_on_target_home": _safe_get(stats, "shots_on_target_home", 0),
        "shots_on_target_away": _safe_get(stats, "shots_on_target_away", 0),
        "shots_home": _safe_get(stats, "shots_home", 0),
        "shots_away": _safe_get(stats, "shots_away", 0),
        "corners_home": _safe_get(stats, "corners_home", 0),
        "corners_away": _safe_get(stats, "corners_away", 0),
        "dangerous_attacks_home": _safe_get(stats, "dangerous_attacks_home", 0),
        "dangerous_attacks_away": _safe_get(stats, "dangerous_attacks_away", 0),
        "possession_home": _safe_get(stats, "possession_home", 50),
        "possession_away": _safe_get(stats, "possession_away", 50),
        "red_cards_home": _safe_get(stats, "red_cards_home", 0),
        "red_cards_away": _safe_get(stats, "red_cards_away", 0),
        "over_1_5_goals": _safe_get(odds, "over_1_5_goals", 2.0),
        "over_2_5_goals": _safe_get(odds, "over_2_5_goals", 2.0),
        "under_2_5_goals": _safe_get(odds, "under_2_5_goals", 2.0),
        "btts_yes": _safe_get(odds, "btts_yes", 2.0),
        "btts_no": _safe_get(odds, "btts_no", 2.0),
        "over_9_5_corners": _safe_get(odds, "over_9_5_corners", 2.0),
        "under_9_5_corners": _safe_get(odds, "under_9_5_corners", 2.0),
    }

    feature_vector = [values[name] if values[name] is not None else 0.0 for name in NUMERIC_FEATURES]
    return feature_vector


def encode_context(context: str) -> List[int]:
    return [1 if context == "live" else 0, 1 if context == "prematch" else 0, 1 if context == "combo_leg" else 0]


def encode_sport(sport: str) -> List[int]:
    return [1 if sport == "football" else 0, 1 if sport == "basketball" else 0]


def prepare_features(request: PredictionRequest) -> np.ndarray:
    numeric_features = build_feature_vector(request)
    meta_features = encode_context(request.context.value) + encode_sport(request.sport.value)
    return np.array([numeric_features + meta_features], dtype=float)
