from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from flexcast.models.calibration import apply_offsets, fit_offsets, rolling_calibration
from flexcast.settlement import period_starts


def forecast_days(days: list[date], bias: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Forecasts whose truth is p50 + bias + noise, with a band that is too narrow (±5)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        for t in period_starts(d):
            p50 = 100.0
            rows.append(
                {
                    "target_date": d,
                    "start_time": t,
                    "strategy": "m",
                    "p10": p50 - 5,
                    "p50": p50,
                    "p90": p50 + 5,
                    "actual": p50 + bias + rng.normal(0, 20),
                }
            )
    return pd.DataFrame(rows)


def coverage(f: pd.DataFrame) -> float:
    return float(f["actual"].between(f["p10"], f["p90"]).mean())


def test_offsets_restore_nominal_coverage_in_sample() -> None:
    past = forecast_days([date(2025, 6, 1) + timedelta(days=i) for i in range(60)], bias=7)
    assert coverage(past) < 0.3  # ±5 band against noise sd 20
    fixed = apply_offsets(past, fit_offsets(past))
    assert coverage(fixed) == pytest.approx(0.8, abs=0.01)
    assert (fixed["actual"] < fixed["p50"]).mean() == pytest.approx(0.5, abs=0.01)  # bias removed


def test_offsets_per_hour_block() -> None:
    past = forecast_days([date(2025, 6, 1) + timedelta(days=i) for i in range(30)])
    evening = past["start_time"].dt.tz_convert("Europe/London").dt.hour.between(16, 19)
    past.loc[evening, "actual"] += 80  # the model misses evening spikes
    offsets = fit_offsets(past, hours_per_block=4)
    assert offsets["p90"][4] > offsets["p90"][0] + 50  # block 4 = 16:00-19:59
    fixed = apply_offsets(past, offsets, hours_per_block=4)
    assert coverage(fixed[evening]) == pytest.approx(0.8, abs=0.03)


def test_no_quantile_crossing_after_offsets() -> None:
    past = forecast_days([date(2025, 6, 1) + timedelta(days=i) for i in range(30)])
    offsets = pd.DataFrame({"p10": [50.0], "p50": [0.0], "p90": [-50.0]}, index=pd.Index([0]))
    fixed = apply_offsets(past, offsets)
    assert (fixed["p10"] <= fixed["p50"]).all() and (fixed["p50"] <= fixed["p90"]).all()


def test_too_little_history_means_no_adjustment() -> None:
    past = forecast_days([date(2025, 6, 1)])
    offsets = fit_offsets(past, min_rows=500)
    assert (offsets == 0).all().all()


def test_rolling_calibration_only_uses_days_known_at_decision_time() -> None:
    days = [date(2025, 6, 1) + timedelta(days=i) for i in range(40)]
    f = forecast_days(days, bias=7)
    clean = rolling_calibration(f, "m", "m_cal", window_days=20)
    # Poison every day from 2025-06-30 on: calibrated forecasts up to 2025-07-01 must not change
    # (for target D, the last day known at D-1 09:00 is D-2).
    poisoned = f.copy()
    poisoned.loc[poisoned["target_date"] >= date(2025, 6, 30), "actual"] = 1e6
    dirty = rolling_calibration(poisoned, "m", "m_cal", window_days=20)
    keep = clean["target_date"] <= date(2025, 7, 1)
    cols = ["p10", "p50", "p90"]
    pd.testing.assert_frame_equal(
        clean.loc[keep & (clean.strategy == "m_cal"), cols],
        dirty.loc[keep & (dirty.strategy == "m_cal"), cols],
    )
    assert not clean.loc[clean.strategy == "m_cal", cols].equals(
        dirty.loc[dirty.strategy == "m_cal", cols]
    )  # later days do change, so the test can fail


def test_rolling_calibration_appends_a_strategy_and_improves_coverage() -> None:
    days = [date(2025, 6, 1) + timedelta(days=i) for i in range(60)]
    out = rolling_calibration(forecast_days(days, bias=7), "m", "m_cal", window_days=30)
    assert set(out["strategy"]) == {"m", "m_cal"}
    late = out[out["target_date"] >= date(2025, 7, 10)]
    assert coverage(late[late.strategy == "m"]) < 0.3
    assert coverage(late[late.strategy == "m_cal"]) == pytest.approx(0.8, abs=0.06)


def test_adjust_band_only_leaves_p50_unchanged() -> None:
    past = forecast_days([date(2025, 6, 1) + timedelta(days=i) for i in range(30)], bias=7)
    offsets = fit_offsets(past, adjust=("p10", "p90"))
    assert offsets["p50"][0] == 0.0 and offsets["p90"][0] > 0
    fixed = apply_offsets(past, offsets)
    pd.testing.assert_series_equal(fixed["p50"], past["p50"])


def test_band_crossing_p50_is_clamped_not_reordered() -> None:
    past = forecast_days([date(2025, 6, 1)])
    offsets = pd.DataFrame({"p10": [0.0], "p50": [0.0], "p90": [-50.0]}, index=pd.Index([0]))
    fixed = apply_offsets(past, offsets)
    pd.testing.assert_series_equal(fixed["p50"], past["p50"])  # untouched
    assert (fixed["p90"] == fixed["p50"]).all()
