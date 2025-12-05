from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from app.config import RAW_DATA_PATH
from app.features.extractor import NUMERIC_FEATURES


def generate_rows(n: int) -> List[dict]:
    rng = np.random.default_rng(42)
    rows: List[dict] = []
    for _ in range(n):
        context = rng.choice(["live", "prematch", "combo"])
        sport = rng.choice(["football", "basketball"])
        outcome = rng.choice(["home", "away", "draw"])

        base_stats = rng.normal(5, 2, size=len(NUMERIC_FEATURES)).clip(min=0)
        row = {name: float(value) for name, value in zip(NUMERIC_FEATURES, base_stats)}
        row.update(
            {
                "context_live": 1 if context == "live" else 0,
                "context_prematch": 1 if context == "prematch" else 0,
                "context_combo": 1 if context == "combo" else 0,
                "sport_football": 1 if sport == "football" else 0,
                "sport_basketball": 1 if sport == "basketball" else 0,
                "target_outcome": outcome,
                "context": context,
                "sport": sport,
                "fixture_id": rng.integers(1_000_000),
                "league_id": rng.integers(500),
            }
        )
        rows.append(row)
    return rows


def main(path: Path = RAW_DATA_PATH, samples: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(samples)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Sample dataset written to {path}")


if __name__ == "__main__":
    main()
