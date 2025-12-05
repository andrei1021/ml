from fastapi import FastAPI

from app.config import API_PREFIX, DEFAULT_MODEL_PATH
from app.data.schema import PredictionRequest, PredictionResponse, PredictionResult
from app.features.extractor import prepare_features
from app.models.classifier import MatchOutcomeModel

app = FastAPI(title="Match Prediction Service", version="0.1.0")
model = MatchOutcomeModel(DEFAULT_MODEL_PATH)
model.load_or_create()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model.pipeline is not None}


@app.post(f"{API_PREFIX}/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    features = prepare_features(payload)
    probability_home, probability_away, probability_draw = model.predict_proba(features)

    suggested_market = ""
    if probability_home > max(probability_away, probability_draw):
        suggested_market = "Home win"
    elif probability_away > max(probability_home, probability_draw):
        suggested_market = "Away win"
    else:
        suggested_market = "Draw or double chance"

    rationale = "Model suggestion based on provided stats and odds."

    prediction = PredictionResult(
        probability_home_win=probability_home,
        probability_away_win=probability_away,
        probability_draw=probability_draw,
        suggested_market=suggested_market,
        rationale=rationale,
    )

    return PredictionResponse(
        fixture_id=payload.fixture_id,
        league_id=payload.league_id,
        context=payload.context,
        prediction=prediction,
    )


@app.get(f"{API_PREFIX}/model/info")
def model_info() -> dict:
    exists = DEFAULT_MODEL_PATH.exists()
    return {"model_path": str(DEFAULT_MODEL_PATH), "available": exists}
