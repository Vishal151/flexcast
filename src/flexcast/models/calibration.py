"""Conformal calibration of quantile forecasts from their own recent track record.

For each quantile q, the offset is the q-quantile of (actual - forecast_q) over past out-of-sample
forecasts, so that after shifting, a fraction q of past actuals fall at or below the forecast. Done
separately per block of London hours (the model is wrong in different ways in the evening than
overnight). Offsets are refitted for every target day from a rolling window of days whose prices
were all known at that day's decision time, so it behaves the same in the backtest and in the
daily plan job.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from flexcast.backtest import BacktestResult, training_cutoff, valuate
from flexcast.features import decision_time_for
from flexcast.models.base import COLUMNS, QUANTILES
from flexcast.optimise import Battery
from flexcast.settlement import LONDON


def _block(start_time: pd.Series, hours_per_block: int | None) -> pd.Series:
    if hours_per_block is None:
        return pd.Series(0, index=start_time.index)
    blocks: pd.Series = start_time.dt.tz_convert(LONDON).dt.hour // hours_per_block
    return blocks


def fit_offsets(
    past: pd.DataFrame,
    hours_per_block: int | None = None,
    min_rows: int = 200,
    adjust: tuple[str, ...] = tuple(COLUMNS),
) -> pd.DataFrame:
    """Offsets (index: hour block, columns p10/p50/p90) from past forecasts with actuals.

    Blocks with fewer than `min_rows` observations get no adjustment, and neither do quantiles
    not listed in `adjust` (e.g. adjust=("p10", "p90") widens the band but leaves p50, and so
    the battery's decisions, unchanged).
    """
    blocks = _block(past["start_time"], hours_per_block)
    all_blocks = range(24 // hours_per_block) if hours_per_block else range(1)
    rows = {}
    for b in all_blocks:
        g = past[blocks == b]
        if len(g) < min_rows:
            rows[b] = dict.fromkeys(COLUMNS, 0.0)
            continue
        rows[b] = {
            col: float(np.quantile(g["actual"] - g[col], q)) if col in adjust else 0.0
            for q, col in zip(QUANTILES, COLUMNS, strict=True)
        }
    return pd.DataFrame.from_dict(rows, orient="index")[COLUMNS]


def apply_offsets(
    pred: pd.DataFrame, offsets: pd.DataFrame, hours_per_block: int | None = None
) -> pd.DataFrame:
    """Shift each quantile by its block's offset. If the band would cross p50, it is clamped to
    p50 (not re-sorted), so p50 only ever moves when it was itself adjusted."""
    shift = offsets.reindex(_block(pred["start_time"], hours_per_block)).fillna(0.0)
    p10, p50, p90 = (pred[c].to_numpy() + shift[c].to_numpy() for c in COLUMNS)
    return pred.assign(p10=np.minimum(p10, p50), p50=p50, p90=np.maximum(p90, p50))


def rolling_calibration(
    forecasts: pd.DataFrame,
    strategy: str,
    name: str,
    window_days: int,
    hours_per_block: int | None = None,
    adjust: tuple[str, ...] = tuple(COLUMNS),
) -> pd.DataFrame:
    """Append `name`: `strategy` recalibrated each day from the previous `window_days` of its own
    forecasts whose actuals were known at that day's decision time (target D: up to D-2)."""
    base = forecasts[forecasts["strategy"] == strategy]
    out = []
    for d, day in base.groupby("target_date", sort=True):
        assert isinstance(d, date)
        last_known = training_cutoff(decision_time_for(d))
        first = last_known - timedelta(days=window_days - 1)
        past = base[(base["target_date"] >= first) & (base["target_date"] <= last_known)]
        offsets = fit_offsets(past, hours_per_block, adjust=adjust)
        out.append(apply_offsets(day, offsets, hours_per_block).assign(strategy=name))
    return pd.concat([forecasts, *out], ignore_index=True)


def add_calibrated(
    result: BacktestResult,
    battery: Battery,
    base: str,
    name: str,
    window_days: int,
    hours_per_block: int | None,
    adjust: tuple[str, ...],
) -> None:
    """Add `name` (calibrated `base`) to both the validation and the test forecasts, so later
    steps (e.g. choosing the risk-aware alpha on validation) can use it. Each day's offsets come
    only from days known at its decision time, so validation days calibrate test days."""
    test_start = result.forecasts["target_date"].min()
    every = pd.concat([result.validation_forecasts, result.forecasts], ignore_index=True)
    calibrated = rolling_calibration(every, base, name, window_days, hours_per_block, adjust)
    rows = calibrated[calibrated["strategy"] == name]
    is_test = rows["target_date"] >= test_start
    result.validation_forecasts = pd.concat(
        [result.validation_forecasts, rows[~is_test]], ignore_index=True
    )
    result.forecasts = pd.concat([result.forecasts, rows[is_test]], ignore_index=True)
    result.daily = pd.concat([result.daily, valuate(rows[is_test], battery)], ignore_index=True)
