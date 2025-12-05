# Match Prediction Microservice

FastAPI microservice that exposes machine learning predictions for football and basketball fixtures. The service is designed to be called from a Discord bot via JSON requests.

## Features
- JSON contract compatible with the Node.js bot (live, prematch, combo leg contexts).
- Feature extraction pipeline to convert match payloads into ML-ready vectors.
- Training scripts with a lightweight NumPy-based softmax classifier (no compiled scikit-learn dependency).
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

## Environment prerequisites
- **Python**: 3.10–3.14 are supported. The model code now uses only NumPy/Joblib for training and serving, so no compiled scikit-learn wheels are required.
- **Pip**: use `python -m pip` if `pip` is not on your PATH.
- **Binary wheels first**: prefer binary wheels to avoid compilation on Windows: `python -m pip install --upgrade pip setuptools wheel` followed by `python -m pip install --prefer-binary -r requirements.txt`.

## Getting started
1. Install dependencies (works on Python 3.10–3.14):
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install --prefer-binary -r requirements.txt
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
- The default model is a lightweight softmax regression implemented with NumPy so it can be trained and served on Python 3.14 without C-extension builds. Swap in a different estimator inside `app/models/classifier.py` if you prefer.
- Dataset columns are defined in `app/features/extractor.py` (features) and `scripts/prepare_dataset_example.py` (one-hot context/sport indicators).
- Extend `PredictionResult` to include additional markets as needed.

## Troubleshooting
- **Binary wheels not found**: ensure you are using `python -m pip install --prefer-binary -r requirements.txt` to avoid compiling dependencies from source on Windows.
- **Missing `numpy` or other modules**: confirm dependency installation before running scripts like `prepare_dataset_example.py`.
