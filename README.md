# Match Prediction Microservice

FastAPI microservice that exposes machine learning predictions for football and basketball fixtures. The service is designed to be called from a Discord bot via JSON requests.

## Features
- JSON contract compatible with the Node.js bot (live, prematch, combo leg contexts).
- Feature extraction pipeline to convert match payloads into ML-ready vectors.
- Training scripts with scikit-learn to retrain and save models.
- Simple storage helpers for CSV-based datasets.

## Project layout
```
app/
  main.py                # FastAPI entrypoint
  config.py              # Paths for data/model artifacts
  models/classifier.py   # Model wrapper and loading/saving helpers
  features/extractor.py  # JSON -> numeric feature conversion
  data/schema.py         # Pydantic request/response schemas
  data/storage.py        # CSV storage helpers
scripts/
  train_model.py         # Train model from CSV dataset
  prepare_dataset_example.py # Generate example dataset
models_store/            # Saved models (created at runtime)
data_store/              # CSV datasets (created at runtime)
requirements.txt
```

## Getting started
1. Install dependencies (use `python -m pip` if `pip` is not on your PATH):
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Generate an example dataset:
   ```bash
   python scripts/prepare_dataset_example.py
   ```
3. Train a model:
   ```bash
   python scripts/train_model.py --dataset data_store/training_data.csv --output models_store/match_predictor.joblib
   ```
4. Run the API locally:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### Example request
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "context": "live",
    "sport": "football",
    "fixture_id": 12345,
    "league_id": 283,
    "minute": 67,
    "score": {"home": 1, "away": 1},
    "stats": {
      "xg_home": 1.8,
      "xg_away": 0.6,
      "shots_on_target_home": 7,
      "shots_on_target_away": 2,
      "shots_home": 12,
      "shots_away": 5,
      "corners_home": 5,
      "corners_away": 3,
      "dangerous_attacks_home": 40,
      "dangerous_attacks_away": 22,
      "possession_home": 58,
      "possession_away": 42,
      "red_cards_home": 0,
      "red_cards_away": 0
    },
    "odds": {
      "over_1_5_goals": 1.30,
      "over_2_5_goals": 1.85,
      "under_2_5_goals": 1.95,
      "btts_yes": 1.70,
      "btts_no": 2.10,
      "over_9_5_corners": 1.90,
      "under_9_5_corners": 1.90
    }
  }'
```

## Notes
- The initial model uses Gradient Boosting with isotonic calibration. Replace the estimator in `MatchOutcomeModel.create_default_pipeline` for other algorithms.
- Dataset columns are defined in `app/features/extractor.py` (features) and `scripts/prepare_dataset_example.py` (one-hot context/sport indicators).
- Extend `PredictionResult` to include additional markets as needed.
