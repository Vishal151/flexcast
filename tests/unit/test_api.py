from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flexcast.api import create_app
from flexcast.config import Settings
from flexcast.history import History
from flexcast.jobs import plan, train
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
def client(storage: Storage, history: History) -> TestClient:
    return TestClient(create_app(storage, lambda: history))


@pytest.fixture
def planned(history: History, storage: Storage) -> None:
    train(
        history,
        storage,
        as_of=datetime(2025, 6, 24, 12, tzinfo=UTC),
        git_sha="t",
        market="day_ahead",
    )
    plan(history, storage, TARGET, Settings())


def test_health_before_any_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "latest_model": None}


def test_health_reports_latest_model(client: TestClient, planned: None) -> None:
    assert (
        client.get("/health").json()["latest_model"].startswith("models/day_ahead/lgbm_quantile/")
    )


def test_schedule(client: TestClient, planned: None) -> None:
    r = client.get("/schedule", params={"date": str(TARGET)})
    assert r.status_code == 200
    body = r.json()
    assert body["target_date"] == str(TARGET)
    assert body["decision_time"] == "2025-06-24T08:00:00Z"
    assert len(body["periods"]) == 48
    first = body["periods"][0]
    assert set(first) == {"start_time", "charge_kw", "discharge_kw", "soc_kwh"}
    assert first["start_time"] == "2025-06-24T23:00:00Z"


def test_forecast(client: TestClient, planned: None) -> None:
    body = client.get("/forecast", params={"date": str(TARGET)}).json()
    assert len(body["periods"]) == 48
    p = body["periods"][0]
    assert p["p10"] <= p["p50"] <= p["p90"]


@pytest.mark.parametrize("path", ["/schedule", "/forecast"])
def test_unplanned_date_is_404(client: TestClient, path: str) -> None:
    r = client.get(path, params={"date": "2030-01-01"})
    assert r.status_code == 404
    assert "2030-01-01" in r.json()["detail"]


def test_bad_date_is_422(client: TestClient) -> None:
    assert client.get("/schedule", params={"date": "tomorrow"}).status_code == 422


def test_metrics(client: TestClient, planned: None) -> None:
    body = client.get("/metrics", params={"end": "2025-06-26"}).json()
    assert body["days"] == 1 and body["from"] == str(TARGET)
    assert body["perfect_gbp"] >= body["settled_gbp"]


def test_openapi_docs_available(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200


def test_metrics_on_a_fresh_empty_bucket(storage: Storage) -> None:
    """Regression: /metrics returned 500 when no data had been ingested yet."""
    client = TestClient(create_app(storage))  # default loader: History.load(storage)
    body = client.get("/metrics").json()
    assert body["days"] == 0 and body["pct_of_perfect"] is None
