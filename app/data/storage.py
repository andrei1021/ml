from pathlib import Path
from typing import Iterable, List

import pandas as pd

from app.config import RAW_DATA_PATH


COLUMNS = [
    "context",
    "sport",
    "fixture_id",
    "league_id",
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
    "target_outcome",
]


def ensure_storage(path: Path = RAW_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)


def append_records(records: Iterable[dict], path: Path = RAW_DATA_PATH) -> None:
    ensure_storage(path)
    df_new = pd.DataFrame(records)
    df_new.to_csv(path, mode="a", header=False, index=False)


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    ensure_storage(path)
    return pd.read_csv(path)


def list_datasets() -> List[Path]:
    return sorted(RAW_DATA_PATH.parent.glob("*.csv"))
