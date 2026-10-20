from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import ClassVar, Self

import pandas as pd
import pytest

from flexcast.backtest import (
    PERFECT,
    BacktestConfig,
    alpha_grid,
    run,
    select_alpha,
    summarise,
    training_cutoff,
    valuate,
    with_risk_aware,
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
    validation_days=3,  # forecasting starts 2025-06-17
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
    retrain_days = [date(2025, 6, 17), date(2025, 6, 20), date(2025, 6, 23), date(2025, 6, 26)]
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


# --- validation window and risk-aware scheduling ----------------------------------------------


def test_validation_days_are_forecast_but_kept_out_of_test_results(history: History) -> None:
    res = run(history, CFG, {"naive": NaiveLastDay})
    val_days = sorted(set(res.validation_forecasts["target_date"]))
    assert val_days == [date(2025, 6, 17), date(2025, 6, 18), date(2025, 6, 19)]
    assert res.forecasts["target_date"].min() == CFG.test_start
    assert res.daily["target_date"].min() == CFG.test_start


def anti_forecast(forecasts: pd.DataFrame, band: float) -> pd.DataFrame:
    """A deliberately bad strategy: each day's actual shape reversed, with a +-band interval."""
    perfect = forecasts[forecasts["strategy"] == PERFECT].copy()
    reversed_p50 = perfect.groupby("target_date")["actual"].transform(lambda s: s[::-1].to_numpy())
    bad = perfect.assign(
        strategy="bad", p50=reversed_p50, p10=reversed_p50 - band, p90=reversed_p50 + band
    )
    return pd.concat([forecasts, bad], ignore_index=True)


def test_risk_aware_alpha_zero_equals_p50_strategy(history: History) -> None:
    f = anti_forecast(run(history, CFG, {"naive": NaiveLastDay}).forecasts, band=500)
    daily = valuate(with_risk_aware(f, "bad", 0.0, "bad_risk"), CFG.battery)
    wide = daily.pivot(index="target_date", columns="strategy", values="value_gbp")
    pd.testing.assert_series_equal(wide["bad"], wide["bad_risk"], check_names=False)


def test_wide_uncertainty_stops_a_bad_forecast_trading(history: History) -> None:
    f = anti_forecast(run(history, CFG, {"naive": NaiveLastDay}).forecasts, band=500)
    s = summarise(valuate(with_risk_aware(f, "bad", 1.0, "bad_risk"), CFG.battery))
    assert s["gbp_per_year"]["bad"] < 0  # trading on a reversed shape loses money
    assert s.loc["bad_risk", "gbp_per_year"] == pytest.approx(0, abs=1e-6)  # so it sits out


def test_select_alpha_picks_best_validation_alpha(history: History) -> None:
    val = anti_forecast(run(history, CFG, {"naive": NaiveLastDay}).validation_forecasts, band=500)
    alphas = (0.0, 0.25, 0.5, 1.0)
    alpha, grid = select_alpha(val, CFG.battery, "bad", alphas)
    assert list(grid.index) == ["alpha=0", "alpha=0.25", "alpha=0.5", "alpha=1"]
    assert alpha == 0.25  # first alpha that stops the losses; ties go to the smallest
    gbp = grid["gbp_per_year"]
    assert float(gbp["alpha=0"]) < float(gbp["alpha=0.25"])
    pd.testing.assert_frame_equal(grid, alpha_grid(val, CFG.battery, "bad", alphas))
