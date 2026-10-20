"""LightGBM quantile regression: one booster per quantile (p10, p50, p90)."""

import json
from typing import Any, ClassVar, Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from flexcast.features import ALL_COLUMNS, FEATURE_COLUMNS
from flexcast.models.base import COLUMNS, QUANTILES

PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
NUM_ROUNDS = 300


class LGBMQuantile:
    name: ClassVar[str] = "lgbm_quantile"

    def __init__(
        self, boosters: dict[str, lgb.Booster] | None = None, features: list[str] | None = None
    ) -> None:
        self.boosters = boosters or {}
        # Which input columns to use. Default: everything adopted so far (solar added in
        # Stage 2c checkpoint B after the validation-window comparison in notebook 03).
        self.features = list(features) if features is not None else list(ALL_COLUMNS)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        data = lgb.Dataset(X[self.features], label=y, free_raw_data=False)
        self.boosters = {
            col: lgb.train({**PARAMS, "alpha": q}, data, num_boost_round=NUM_ROUNDS)
            for q, col in zip(QUANTILES, COLUMNS, strict=True)
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.boosters:
            raise RuntimeError("model is not fitted")
        raw = np.column_stack([self.boosters[c].predict(X[self.features]) for c in COLUMNS])
        # Separately trained quantiles can cross; sorting each row restores p10 <= p50 <= p90.
        return pd.DataFrame(np.sort(raw, axis=1), columns=COLUMNS, index=X.index)

    def to_files(self) -> dict[str, str]:
        files = {f"{c}.txt": b.model_to_string() for c, b in self.boosters.items()}
        params = {**PARAMS, "num_rounds": NUM_ROUNDS, "features": self.features}
        files["params.json"] = json.dumps(params, sort_keys=True) + "\n"
        return files

    @classmethod
    def from_files(cls, files: dict[str, str]) -> Self:
        # Models saved before Stage 2c didn't record features; they used the original 23.
        features = json.loads(files["params.json"]).get("features", FEATURE_COLUMNS)
        return cls({c: lgb.Booster(model_str=files[f"{c}.txt"]) for c in COLUMNS}, features)
