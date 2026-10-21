"""Point-in-time feature build: one row per settlement period of target_date, using only data
knowable before decision_time (via History.as_of).

"Same time on another day" matches London clock time (HH:MM), not settlement period number, so
46/50-period days don't shift the daily shape by an hour. NDF is stitched per period: for each
target period, the latest vintage published before the decision (docs/data-sources.md).
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import cache

import holidays
import numpy as np
import pandas as pd

from flexcast.history import History
from flexcast.settlement import LONDON, london_day_bounds, period_starts

FEATURE_COLUMNS = [
    "settlement_period",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "n_periods",
    "price_d2",
    "price_d7",
    "price_d2_mean",
    "price_d2_min",
    "price_d2_max",
    "price_roll7_mean",
    "price_roll7_std",
    "price_latest",
    "price_latest_age_h",
    "mid_d2",
    "mid_latest",
    "ndf",
    "ndf_age_h",
    "wind",
    "wind_age_h",
    "residual_demand",
]
# Stage 2c checkpoint B: NESO embedded (distribution-connected) solar and wind forecasts.
EMBEDDED_COLUMNS = [
    "solar",
    "solar_capacity_factor",
    "embedded_wind",
    "embedded_age_h",
]
ALL_COLUMNS = FEATURE_COLUMNS + EMBEDDED_COLUMNS
# Stage 2c checkpoint D: day-ahead auction prices. Yesterday's are published at 10:00 on D-2,
# so at the D-1 09:00 decision the latest complete day is D-1 (not D-2 as for imbalance prices).
DAY_AHEAD_COLUMNS = [
    "da_d1",
    "da_d7",
    "da_d1_mean",
    "da_d1_min",
    "da_d1_max",
    "da_roll7_mean",
]
# Candidate inputs (docs/candidate-sources.md), built always and adopted only on validation.
# Gas: System Average Price in £/MWh; at D-1 09:00 the latest gas day known is D-3.
GAS_COLUMNS = ["gas_price", "gas_price_7d"]
# Plant availability for day D (Elexon FOU2T14D, latest vintage, published by D-2), MW.
# margin = what could run (dispatchable + interconnector imports + forecast wind) minus demand.
AVAILABILITY_COLUMNS = [
    "avail_ccgt",
    "avail_nuclear",
    "avail_interconnector",
    "avail_dispatchable",
    "margin",
]
EXTRA_FEATURES = {"gas": GAS_COLUMNS, "availability": AVAILABILITY_COLUMNS}
OUTPUT_COLUMNS = ALL_COLUMNS + DAY_AHEAD_COLUMNS + GAS_COLUMNS + AVAILABILITY_COLUMNS
DAY_AHEAD_MODEL_FEATURES = ALL_COLUMNS + DAY_AHEAD_COLUMNS


@dataclass(frozen=True)
class Features:
    X: pd.DataFrame  # index: UTC period start; columns: FEATURE_COLUMNS
    provenance: dict[str, pd.Timestamp]  # source -> latest available_at among rows used
    decision_time: datetime
    target_date: date


def decision_time_for(target_date: date, at: time = time(9)) -> datetime:
    """D-1 at a London wall-clock time (default 09:00), as a tz-aware instant."""
    return datetime.combine(target_date - timedelta(days=1), at, LONDON)


@cache
def _holidays() -> holidays.HolidayBase:
    return holidays.country_holidays("GB", subdiv="ENG")


class _Provenance:
    def __init__(self) -> None:
        self.latest: dict[str, pd.Timestamp] = {}

    def use(self, source: str, rows: pd.DataFrame) -> pd.DataFrame:
        if not rows.empty:
            t = rows["available_at"].max()
            self.latest[source] = max(t, self.latest.get(source, t))
        return rows


def _same_time(df: pd.DataFrame, day: date, prov: _Provenance, source: str) -> pd.Series:
    """Values on London `day`, keyed by HH:MM (first occurrence on an autumn DST day)."""
    rows = prov.use(source, df[df["london_date"] == day])
    return rows.drop_duplicates("hhmm").set_index("hhmm")["price"]


def _age_hours(decision: pd.Timestamp, published: pd.Series) -> np.ndarray:
    age: np.ndarray = ((decision - published).dt.total_seconds() / 3600).to_numpy(dtype=float)
    return age


def _latest_per_start(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values("publish_time").drop_duplicates("start_time", keep="last")


def build(history: History, decision_time: datetime, target_date: date) -> Features:
    day_start, day_end = london_day_bounds(target_date)
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be tz-aware")
    if decision_time >= day_start:
        raise ValueError(f"decision_time {decision_time} is not before target day {target_date}")

    starts = pd.DatetimeIndex(period_starts(target_date), name="start_time")
    local = starts.tz_convert(LONDON)
    hhmm = pd.Index(local.strftime("%H:%M"))
    prov = _Provenance()
    decision = pd.Timestamp(decision_time)
    hours = pd.Timedelta(hours=1)

    X = pd.DataFrame(index=starts)
    X["settlement_period"] = np.arange(1, len(starts) + 1)
    X["hour"] = local.hour + local.minute / 60
    X["dow"] = local.dayofweek
    X["month"] = local.month
    X["is_weekend"] = (local.dayofweek >= 5).astype(int)
    X["is_holiday"] = int(target_date in _holidays())
    X["n_periods"] = len(starts)

    # Prices and MID: lagged by whole London days, matched on clock time.
    prices = history.as_of("prices", decision_time)
    d2 = _same_time(prices, target_date - timedelta(days=2), prov, "prices")
    X["price_d2"] = hhmm.map(d2).to_numpy(dtype=float)
    X["price_d7"] = hhmm.map(
        _same_time(prices, target_date - timedelta(days=7), prov, "prices")
    ).to_numpy(dtype=float)
    X["price_d2_mean"] = d2.mean()
    X["price_d2_min"] = d2.min()
    X["price_d2_max"] = d2.max()
    week = [target_date - timedelta(days=k) for k in range(2, 9)]
    roll = prov.use("prices", prices[prices["london_date"].isin(week)]).groupby("hhmm")["price"]
    X["price_roll7_mean"] = hhmm.map(roll.mean()).to_numpy(dtype=float)
    X["price_roll7_std"] = hhmm.map(roll.std()).to_numpy(dtype=float)
    latest = prov.use("prices", prices.tail(1))
    X["price_latest"] = latest["price"].iloc[0] if len(latest) else np.nan
    X["price_latest_age_h"] = (
        (decision - latest["start_time"].iloc[0]) / hours if len(latest) else np.nan
    )

    mid = history.as_of("mid", decision_time)
    X["mid_d2"] = hhmm.map(_same_time(mid, target_date - timedelta(days=2), prov, "mid")).to_numpy(
        dtype=float
    )
    mid_latest = prov.use("mid", mid.tail(1))
    X["mid_latest"] = mid_latest["price"].iloc[0] if len(mid_latest) else np.nan

    # NDF: per-period latest vintage (stitches the rolling and 07:45 runs).
    ndf = prov.use("ndf", _latest_per_start(history.as_of("ndf", decision_time, target_date)))
    ndf = ndf.set_index("start_time").reindex(starts)
    X["ndf"] = ndf["demand"].to_numpy(dtype=float)
    X["ndf_age_h"] = _age_hours(decision, ndf["publish_time"])

    # WINDFOR: hourly; each half-hour takes its hour's value.
    wind_all = history.as_of("windfor", decision_time)
    in_day = (wind_all["start_time"] >= day_start - hours) & (wind_all["start_time"] < day_end)
    wind = prov.use("windfor", _latest_per_start(wind_all[in_day])).set_index("start_time")
    hour_of = starts.floor("h")
    X["wind"] = wind["generation"].reindex(hour_of).to_numpy(dtype=float)
    X["wind_age_h"] = _age_hours(decision, wind["publish_time"].reindex(hour_of))
    X["residual_demand"] = X["ndf"] - X["wind"]

    # Embedded solar/wind: per-period latest vintage, like NDF.
    emb = history.as_of("embedded", decision_time, target_date)
    emb = prov.use("embedded", _latest_per_start(emb)).set_index("start_time").reindex(starts)
    X["solar"] = emb["solar_mw"].to_numpy(dtype=float)
    X["solar_capacity_factor"] = (emb["solar_mw"] / emb["solar_capacity_mw"]).to_numpy(dtype=float)
    X["embedded_wind"] = emb["wind_mw"].to_numpy(dtype=float)
    X["embedded_age_h"] = _age_hours(decision, emb["publish_time"])

    # Day-ahead auction prices, matched on London clock time like the imbalance lags.
    da = history.as_of("day_ahead", decision_time)
    da_d1 = _same_time(da, target_date - timedelta(days=1), prov, "day_ahead")
    X["da_d1"] = hhmm.map(da_d1).to_numpy(dtype=float)
    X["da_d7"] = hhmm.map(
        _same_time(da, target_date - timedelta(days=7), prov, "day_ahead")
    ).to_numpy(dtype=float)
    X["da_d1_mean"] = da_d1.mean()
    X["da_d1_min"] = da_d1.min()
    X["da_d1_max"] = da_d1.max()
    da_week = [target_date - timedelta(days=k) for k in range(1, 8)]
    da_roll = prov.use("day_ahead", da[da["london_date"].isin(da_week)]).groupby("hhmm")["price"]
    X["da_roll7_mean"] = hhmm.map(da_roll.mean()).to_numpy(dtype=float)

    # Gas: the same value for every period of the day.
    gas = prov.use("gas", history.as_of("gas", decision_time).tail(7))
    X["gas_price"] = gas["price"].iloc[-1] if len(gas) else np.nan
    X["gas_price_7d"] = gas["price"].mean() if len(gas) else np.nan

    # Availability: the latest vintage covering the target day, summed by kind of plant.
    av = history.as_of("availability", decision_time, target_date)
    av = prov.use("availability", av[av["publish_time"] == av["publish_time"].max()])
    by_fuel = av.groupby("fuel_type")["output_usable"].sum()
    ic = av.loc[av["interconnector"], "output_usable"].sum()
    disp = av.loc[~av["interconnector"] & (av["fuel_type"] != "WIND"), "output_usable"].sum()
    empty = av.empty
    X["avail_ccgt"] = np.nan if empty else by_fuel.get("CCGT", 0)
    X["avail_nuclear"] = np.nan if empty else by_fuel.get("NUCLEAR", 0)
    X["avail_interconnector"] = np.nan if empty else ic
    X["avail_dispatchable"] = np.nan if empty else disp
    X["margin"] = X["avail_dispatchable"] + X["avail_interconnector"] + X["wind"] - X["ndf"]

    return Features(
        X=X[OUTPUT_COLUMNS],
        provenance=prov.latest,
        decision_time=decision_time,
        target_date=target_date,
    )
