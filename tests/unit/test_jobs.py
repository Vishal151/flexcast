import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from flexcast.config import Settings
from flexcast.history import History
from flexcast.jobs import LATEST, load_latest_model, plan, plan_metrics, train
from flexcast.settlement import period_starts
from flexcast.storage import Storage

HistoryFactory = Callable[[date, date], History]
TARGET = date(2025, 6, 25)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(f"file://{tmp_path}")


@pytest.fixture
def history(synthetic_history: HistoryFactory) -> History:
    return synthetic_history(date(2025, 6, 1), date(2025, 6, 27))


@pytest.fixture
def trained(history: History, storage: Storage) -> str:
    return train(history, storage, as_of=datetime(2025, 6, 24, 12, tzinfo=UTC), git_sha="abc123")


def test_train_saves_model_with_metadata_and_latest_pointer(trained: str, storage: Storage) -> None:
    # At 12:00Z on 24 June every price of 23 June is known (last one at 23:30Z on the 23rd).
    assert trained == "models/lgbm_quantile/train_end=2025-06-23"
    assert storage.read_text(LATEST).strip() == trained
    meta = json.loads(storage.read_text(f"{trained}/metadata.json"))
    assert meta["git_sha"] == "abc123"
    assert meta["train_window"] == ["2025-06-01", "2025-06-23"]
    assert meta["rows"] == 23 * 48
    assert load_latest_model(storage)[1] == trained


def test_train_without_complete_days_fails_clearly(history: History, storage: Storage) -> None:
    with pytest.raises(ValueError, match="no complete days"):
        train(history, storage, as_of=datetime(2025, 5, 1, tzinfo=UTC), git_sha="x")


def test_plan_without_a_model_fails_clearly(history: History, storage: Storage) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        plan(history, storage, TARGET, Settings())


def test_plan_writes_forecast_schedule_and_manifest(
    history: History, storage: Storage, trained: str
) -> None:
    out = plan(history, storage, TARGET, Settings())
    prefix = f"plans/date={TARGET}"
    forecast = storage.read_parquet(f"{prefix}/forecast.parquet")
    schedule = storage.read_parquet(f"{prefix}/schedule.parquet")
    meta = json.loads(storage.read_text(f"{prefix}/plan.json"))
    assert len(forecast) == len(schedule) == len(period_starts(TARGET))
    assert {"start_time", "p10", "p50", "p90"} <= set(forecast.columns)
    assert {"start_time", "charge_kw", "discharge_kw", "soc_kwh"} <= set(schedule.columns)
    assert meta["model"] == trained and meta["risk_alpha"] == 0.0
    assert meta["decision_time"] == "2025-06-24T08:00:00+00:00"
    assert meta["expected_gbp"] == pytest.approx(out.expected_gbp)


def test_plan_risk_alpha_changes_planning_prices(
    history: History, storage: Storage, trained: str
) -> None:
    cautious = plan(history, storage, TARGET, Settings(risk_alpha=1.0))
    plain = plan(history, storage, TARGET, Settings(risk_alpha=0.0))
    assert cautious.expected_gbp <= plain.expected_gbp + 1e-9


def test_plan_is_idempotent(
    history: History, storage: Storage, trained: str, tmp_path: Path
) -> None:
    plan(history, storage, TARGET, Settings())
    files = sorted(p for p in (tmp_path / f"plans/date={TARGET}").iterdir())
    first = {p.name: p.read_bytes() for p in files if p.suffix == ".parquet"}
    plan(history, storage, TARGET, Settings())
    assert {p.name: p.read_bytes() for p in files if p.suffix == ".parquet"} == first


def test_metrics_compare_plans_with_actuals(
    history: History, storage: Storage, trained: str
) -> None:
    for d in (date(2025, 6, 24), date(2025, 6, 25)):
        plan(history, storage, d, Settings())
    m = plan_metrics(history, storage, end=date(2025, 6, 26), days=30)
    assert m["days"] == 2
    assert m["perfect_gbp"] >= m["settled_gbp"]
    assert m["mae"] > 0
    assert m["from"] == "2025-06-24" and m["to"] == "2025-06-25"


def test_metrics_with_no_plans(history: History, storage: Storage) -> None:
    m = plan_metrics(history, storage, end=date(2025, 6, 26), days=30)
    assert m["days"] == 0 and m["pct_of_perfect"] is None


def test_schedule_index_is_utc(history: History, storage: Storage, trained: str) -> None:
    plan(history, storage, TARGET, Settings())
    s = storage.read_parquet(f"plans/date={TARGET}/schedule.parquet")
    assert str(pd.DatetimeIndex(s["start_time"]).tz) == "UTC"


def test_plan_calibrates_from_its_own_past_plans(
    history: History, storage: Storage, trained: str
) -> None:
    # Track record: plans for 18..23 June, whose prices are all known by 25 June's decision.
    for d in (date(2025, 6, 18) + timedelta(days=i) for i in range(6)):
        plan(history, storage, d, Settings())
    settings = Settings(calibration_window_days=10, calibration_adjust="p10,p90")
    out = plan(history, storage, TARGET, settings)
    meta = json.loads(storage.read_text(f"plans/date={TARGET}/plan.json"))
    assert set(meta["calibration_offsets"]) == {"0"}
    f = out.forecast
    assert {"raw_p10", "raw_p50", "raw_p90"} <= set(f.columns)
    assert (f["p50"] == f["raw_p50"]).all()  # band-only: decisions unchanged
    assert not (f["p90"] == f["raw_p90"]).all()  # but the band moved


def test_plan_without_track_record_is_uncalibrated(
    history: History, storage: Storage, trained: str
) -> None:
    out = plan(history, storage, TARGET, Settings(calibration_window_days=10))
    assert (out.forecast["p90"] == out.forecast["raw_p90"]).all()
    meta = json.loads(storage.read_text(f"plans/date={TARGET}/plan.json"))
    assert meta["calibration_offsets"] == {}
