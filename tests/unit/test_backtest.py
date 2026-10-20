from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import ClassVar, Self

import pandas as pd
import pytest

from flexcast.backtest import (
    PERFECT,
    BacktestConfig,
    run,
    summarise,
    training_cutoff,
    valuate,
    worst_days,
)
from flexcast.history import History
from flexcast.models.baselines import NaiveLastDay, SeasonalNaive

HistoryFactory = Callable[[date, date], History]

CFG = BacktestConfig(
    test_start=date(2025, 6, 20),
    test_end=date(2025, 6, 26),
    train_start=date(2025, 6, 12),
    retrain_every_days=3,
)


@pytest.fixture
def history(synthetic_history: HistoryFactory) -> History:
    return synthetic_history(date(2025, 6, 1), date(2025, 6, 27))


class SpyModel:
    """Records the training days it was given; predicts the training mean."""

    name: ClassVar[str] = "spy"
    fits: ClassVar[list[tuple[date, date]]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        days = X.index.get_level_values("target_date")
        SpyModel.fits.append((days.min(), days.max()))
        self.mean = float(y.mean())

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self.mean for c in ("p10", "p50", "p90")}, index=X.index)

    def to_files(self) -> dict[str, str]:
        return {}

    @classmethod
    def from_files(cls, files: dict[str, str]) -> Self:
        return cls()


def test_training_cutoff_is_d_minus_2_at_9am() -> None:
    summer = datetime(2025, 6, 10, 8, 0, tzinfo=UTC)  # 09:00 BST on D-1 for D = 11 June
    winter = datetime(2025, 1, 14, 9, 0, tzinfo=UTC)  # 09:00 GMT on D-1 for D = 15 Jan
    assert training_cutoff(summer) == date(2025, 6, 9)
    assert training_cutoff(winter) == date(2025, 1, 13)
    # Just after midnight + 30 min + 1 h lag, D-1's... still nothing of D-1 is complete.
    assert training_cutoff(datetime(2025, 6, 9, 23, 31, tzinfo=UTC)) == date(2025, 6, 9)


def test_models_only_train_on_days_known_at_decision_time(history: History) -> None:
    SpyModel.fits.clear()
    run(history, CFG, {"spy": SpyModel})
    retrain_days = [date(2025, 6, 20), date(2025, 6, 23), date(2025, 6, 26)]
    assert len(SpyModel.fits) == len(retrain_days)
    for (first, last), d in zip(SpyModel.fits, retrain_days, strict=True):
        assert first == CFG.train_start
        assert (d - last).days == 2  # D-2 is the last complete day at D-1 09:00


def test_perfect_foresight_is_an_upper_bound(history: History) -> None:
    daily = run(
        history, CFG, {"naive": NaiveLastDay, "seasonal": SeasonalNaive, "spy": SpyModel}
    ).daily
    wide = daily.pivot(index="target_date", columns="strategy", values="value_gbp")
    for s in ("naive", "seasonal", "spy"):
        assert (wide[s] <= wide[PERFECT] + 1e-6).all(), s


def test_forecast_equal_to_actuals_captures_100_percent(history: History) -> None:
    res = run(history, CFG, {"naive": NaiveLastDay})
    oracle = res.forecasts[res.forecasts["strategy"] == PERFECT].assign(strategy="oracle")
    daily = valuate(pd.concat([res.forecasts, oracle]), CFG.battery)
    assert summarise(daily).loc["oracle", "pct_of_perfect"] == pytest.approx(100)
    assert summarise(daily).loc["oracle", "mae"] == 0


def test_every_test_day_has_every_strategy(history: History) -> None:
    res = run(history, CFG, {"naive": NaiveLastDay, "seasonal": SeasonalNaive})
    assert res.daily.groupby("strategy").size().to_dict() == {PERFECT: 7, "naive": 7, "seasonal": 7}
    assert res.train_window == (CFG.train_start, date(2025, 6, 24))


def test_backtest_is_deterministic(history: History) -> None:
    a = run(history, CFG, {"naive": NaiveLastDay, "spy": SpyModel}).daily
    b = run(history, CFG, {"naive": NaiveLastDay, "spy": SpyModel}).daily
    pd.testing.assert_frame_equal(a, b)


def test_summarise_and_worst_days_hand_values() -> None:
    daily = pd.DataFrame(
        {
            "target_date": [date(2025, 1, 1), date(2025, 1, 2)] * 2,
            "strategy": [PERFECT, PERFECT, "m", "m"],
            "value_gbp": [2.0, 4.0, 1.0, 2.0],
            "mae": [0.0, 0.0, 10.0, 20.0],
            "pinball": [0.0, 0.0, 1.0, 3.0],
        }
    )
    s = summarise(daily)
    assert s.loc["m", "pct_of_perfect"] == pytest.approx(50.0)
    assert s.loc["m", "gbp_per_year"] == pytest.approx(1.5 * 365)
    assert s.loc["m", "mae"] == 15.0 and s.loc["m", "pinball"] == 2.0
    w = worst_days(daily, "m", n=1)
    assert list(w.index) == [date(2025, 1, 2)] and w["shortfall_gbp"].iloc[0] == 2.0
