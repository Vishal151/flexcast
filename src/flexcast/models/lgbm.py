"""LightGBM quantile regression: one booster per quantile (p10, p50, p90)."""

import json
from typing import Any, ClassVar, Literal, Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from flexcast.features import ALL_COLUMNS, FEATURE_COLUMNS
from flexcast.models.base import COLUMNS, QUANTILES

# Tuned on the validation window (notebook 04): strong regularisation (big leaves, L2) wins on a
# noisy target. Pre-tuning defaults were num_leaves=31, min_data_in_leaf=50,
# feature_fraction=0.9, lambda_l2=0, 300 rounds.
PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "lambda_l2": 10.0,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
NUM_ROUNDS = 150

# How the central forecast is trained. "quantile" (alpha=0.5) is the median: robust to spikes,
# which also means it ignores them. "regression" (squared error) is the mean, which spikes pull
# up; "huber" is in between.
P50Objective = Literal["quantile", "regression", "huber"]


class LGBMQuantile:
    name: ClassVar[str] = "lgbm_quantile"

    def __init__(
        self,
        boosters: dict[str, lgb.Booster] | None = None,
        features: list[str] | None = None,
        params: dict[str, Any] | None = None,
        num_rounds: int = NUM_ROUNDS,
        p50_objective: P50Objective = "quantile",
    ) -> None:
        self.boosters = boosters or {}
        # Which input columns to use. Default: everything adopted so far (solar added in
        # Stage 2c checkpoint B after the validation-window comparison in notebook 03).
        self.features = list(features) if features is not None else list(ALL_COLUMNS)
        self.params = {**PARAMS, **(params or {})}
        self.num_rounds = num_rounds
        self.p50_objective = p50_objective

    def _params_for(self, q: float) -> dict[str, Any]:
        if q == 0.5 and self.p50_objective != "quantile":
            return {**self.params, "objective": self.p50_objective}
        return {**self.params, "objective": "quantile", "alpha": q}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        data = lgb.Dataset(X[self.features], label=y, free_raw_data=False)
        self.boosters = {
            col: lgb.train(self._params_for(q), data, num_boost_round=self.num_rounds)
            for q, col in zip(QUANTILES, COLUMNS, strict=True)
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.boosters:
            raise RuntimeError("model is not fitted")
        raw = np.column_stack([self.boosters[c].predict(X[self.features]) for c in COLUMNS])
        # Separately trained quantiles can cross; sorting each row restores p10 <= p50 <= p90.
        return pd.DataFrame(np.sort(raw, axis=1), columns=COLUMNS, index=X.index)

    def config(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "num_rounds": self.num_rounds,
            "p50_objective": self.p50_objective,
            "features": self.features,
        }

    def to_files(self) -> dict[str, str]:
        files = {f"{c}.txt": b.model_to_string() for c, b in self.boosters.items()}
        files["params.json"] = json.dumps(self.config(), sort_keys=True) + "\n"
        return files

    @classmethod
    def from_files(cls, files: dict[str, str]) -> Self:
        cfg = json.loads(files["params.json"])
        boosters = {c: lgb.Booster(model_str=files[f"{c}.txt"]) for c in COLUMNS}
        if "params" not in cfg:  # saved before Stage 2c checkpoint C: flat settings
            return cls(boosters, cfg.get("features", FEATURE_COLUMNS))
        return cls(
            boosters, cfg["features"], cfg["params"], cfg["num_rounds"], cfg["p50_objective"]
        )
