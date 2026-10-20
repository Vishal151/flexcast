"""The scheduled jobs' logic: train a model, plan tomorrow, and score past plans.

Storage layout:
  models/lgbm_quantile/train_end=<d>/   model files + metadata.json
  models/lgbm_quantile/LATEST           prefix of the newest model (written last)
  plans/date=<d>/forecast.parquet       p10/p50/p90 per period
  plans/date=<d>/schedule.parquet       charge/discharge/SoC per period
  plans/date=<d>/plan.json              decision time, model, alpha, expected £
"""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from flexcast.backtest import feature_table, training_cutoff
from flexcast.config import Settings
from flexcast.features import build, decision_time_for
from flexcast.history import History
from flexcast.models.base import Model, load_model, save_model
from flexcast.models.registry import MODELS
from flexcast.optimise import Battery, optimise, settle
from flexcast.settlement import period_starts
from flexcast.storage import Storage

MODEL = "lgbm_quantile"
LATEST = f"models/{MODEL}/LATEST"


@dataclass(frozen=True)
class Plan:
    target_date: date
    forecast: pd.DataFrame
    schedule: pd.DataFrame
    expected_gbp: float


def train(history: History, storage: Storage, as_of: datetime, git_sha: str) -> str:
    """Fit on every day whose prices were all known at `as_of`; returns the model's prefix."""
    from flexcast.models.lgbm import LGBMQuantile

    settings = Settings()
    cutoff = training_cutoff(as_of)
    days = sorted(d for d in set(history.prices["london_date"]) if d <= cutoff)
    X, y, _ = feature_table(history, days, settings.decision_at) if days else (None, None, [])
    if X is None or y is None or X.empty:
        raise ValueError(f"no complete days of prices on or before {cutoff} to train on")
    model = LGBMQuantile()
    model.fit(X, y)
    trained = sorted(set(X.index.get_level_values("target_date")))
    prefix = f"models/{MODEL}/train_end={cutoff}"
    save_model(
        storage,
        model,
        prefix,
        {
            "git_sha": git_sha,
            "train_window": [str(trained[0]), str(trained[-1])],
            "rows": len(X),
            "features": model.features,
            "as_of": as_of.isoformat(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    storage.write_text(prefix + "\n", LATEST)  # pointer last: readers never see a partial model
    return prefix


def load_latest_model(storage: Storage) -> tuple[Model, str]:
    if not storage.exists(LATEST):
        raise FileNotFoundError(f"no {LATEST}: run `flexcast train` first")
    prefix = storage.read_text(LATEST).strip()
    return load_model(storage, prefix, MODELS), prefix


def plan(history: History, storage: Storage, target_date: date, settings: Settings) -> Plan:
    """Forecast target_date as of D-1 decision time and optimise the battery on it."""
    model, model_prefix = load_latest_model(storage)
    decision_time = decision_time_for(target_date, settings.decision_at)
    features = build(history, decision_time, target_date)
    pred = model.predict(features.X)
    a = settings.risk_alpha
    buy = pred["p50"] + a * (pred["p90"] - pred["p50"])
    sell = pred["p50"] - a * (pred["p50"] - pred["p10"])
    schedule = optimise(buy, Battery(), sell_prices=sell)

    forecast = pred.rename_axis("start_time").reset_index()
    frame = schedule.frame.rename(columns={"price_gbp_per_mwh": "planning_buy_gbp_per_mwh"})
    frame = frame.assign(planning_sell_gbp_per_mwh=sell.to_numpy()).rename_axis("start_time")
    prefix = f"plans/date={target_date}"
    storage.write_parquet(forecast, f"{prefix}/forecast.parquet")
    storage.write_parquet(frame.reset_index(), f"{prefix}/schedule.parquet")
    meta = {
        "target_date": str(target_date),
        "decision_time": decision_time.astimezone(UTC).isoformat(),
        "model": model_prefix,
        "risk_alpha": a,
        "expected_gbp": schedule.objective_gbp,
        "provenance": {k: v.isoformat() for k, v in features.provenance.items()},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    storage.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", f"{prefix}/plan.json")
    return Plan(target_date, forecast, frame.reset_index(), schedule.objective_gbp)


def planned_dates(storage: Storage) -> list[date]:
    return sorted(
        date.fromisoformat(p.split("date=")[1].split("/")[0])
        for p in storage.list("plans")
        if p.endswith("/plan.json")
    )


def plan_metrics(history: History, storage: Storage, end: date, days: int) -> dict[str, Any]:
    """Score plans for target dates in (end - days, end] that have complete actual prices."""
    battery = Battery()
    rows = []
    for d in planned_dates(storage):
        if not end - timedelta(days=days) < d <= end:
            continue
        actual = history.actual_prices(d)
        if len(actual) != len(period_starts(d)):
            continue
        prefix = f"plans/date={d}"
        forecast = storage.read_parquet(f"{prefix}/forecast.parquet").set_index("start_time")
        schedule = storage.read_parquet(f"{prefix}/schedule.parquet").set_index("start_time")
        rows.append(
            {
                "date": d,
                "mae": float((forecast["p50"] - actual).abs().mean()),
                "settled_gbp": settle(schedule, actual, battery),
                "perfect_gbp": optimise(actual, battery).objective_gbp,
            }
        )
    if not rows:
        return {
            "days": 0,
            "from": None,
            "to": None,
            "mae": None,
            "settled_gbp": 0.0,
            "perfect_gbp": 0.0,
            "pct_of_perfect": None,
        }
    df = pd.DataFrame(rows)
    perfect = float(df["perfect_gbp"].sum())
    settled = float(df["settled_gbp"].sum())
    return {
        "days": len(df),
        "from": str(df["date"].min()),
        "to": str(df["date"].max()),
        "mae": float(df["mae"].mean()),
        "settled_gbp": settled,
        "perfect_gbp": perfect,
        "pct_of_perfect": 100 * settled / perfect if perfect else None,
    }
