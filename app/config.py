from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models_store"
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_PATH = MODEL_DIR / "match_predictor.joblib"
DATA_DIR = BASE_DIR / "data_store"
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_PATH = DATA_DIR / "training_data.csv"

API_PREFIX = "/api"
