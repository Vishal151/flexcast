"""Baselines: copy a recent day's prices. No training; point forecasts (p10 = p50 = p90)."""

import json
from typing import ClassVar, Self

import pandas as pd

from flexcast.models.base import COLUMNS


class _CopyDay:
    name: ClassVar[str]
    source: ClassVar[str]  # feature column holding the lagged price
    fallback: ClassVar[str]  # used where source is NaN (clock time absent on a DST day)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = X[self.source].fillna(X[self.fallback]).fillna(X["price_latest"])
        return pd.DataFrame({c: p for c in COLUMNS}, index=X.index)

    def to_files(self) -> dict[str, str]:
        return {"params.json": json.dumps({"source": self.source}) + "\n"}

    @classmethod
    def from_files(cls, files: dict[str, str]) -> Self:
        return cls()


class NaiveLastDay(_CopyDay):
    """Same London clock time on D-2: the last complete day known at D-1 09:00."""

    name = "naive_last_day"
    source = "price_d2"
    fallback = "price_d2_mean"


class SeasonalNaive(_CopyDay):
    """Same London clock time a week earlier (D-7)."""

    name = "seasonal_naive"
    source = "price_d7"
    fallback = "price_roll7_mean"


class NaiveDayAhead(_CopyDay):
    """Yesterday's day-ahead auction prices (published at 10:00 on D-2)."""

    name = "naive_day_ahead"
    source = "da_d1"
    fallback = "da_d1_mean"


class SeasonalNaiveDayAhead(_CopyDay):
    """Day-ahead auction prices at the same time a week earlier."""

    name = "seasonal_naive_day_ahead"
    source = "da_d7"
    fallback = "da_roll7_mean"
