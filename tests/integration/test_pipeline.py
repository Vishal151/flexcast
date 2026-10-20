"""ingest -> train -> plan in containers; files land in emulated GCS; the API serves them."""

import json
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from flexcast.storage import Storage
from tests.integration.conftest import BUCKET, compose, hashes, ok, run_job, wait_for

pytestmark = pytest.mark.integration

SAMPLE_DAYS = ["2025-03-30", "2025-06-11", "2025-10-26"]  # what upstream-mock has fixtures for
API_PORT = 18080


def ingest_fixtures() -> None:
    for d in SAMPLE_DAYS:
        ok(
            run_job(
                "ingest", "--start", d, "--end", d, "--datasets", "system_prices,mid,fuelhh,agile"
            )
        )
    ok(
        run_job(
            "ingest", "--start", "2025-06-09", "--end", "2025-06-10", "--datasets", "ndf,windfor"
        )
    )


@pytest.fixture(scope="module")
def pipeline(bucket: Storage) -> dict[str, Any]:
    ingest_fixtures()
    first = hashes(bucket, "raw")
    ingest_fixtures()  # re-run: must be a no-op through the real GCS client
    second = hashes(bucket, "raw")
    train_out = ok(run_job("train", "--as-of", "2025-10-28T12:00:00+00:00"))
    plan_out = ok(run_job("plan", "--date", "2025-06-11"))
    return {"first": first, "second": second, "train": train_out, "plan": plan_out}


@pytest.fixture(scope="module")
def api(pipeline: dict[str, Any]) -> Iterator[str]:
    """An API container on the test bucket, on a non-default PORT (as Cloud Run sets it)."""
    name = "flexcast-itest-api"
    compose("rm", "-sf", name)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    ok(
        compose(
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-e",
            f"FLEXCAST_DATA_URI=gs://{BUCKET}",
            "-e",
            "PORT=9090",
            "-p",
            f"{API_PORT}:9090",
            "api",
        )
    )
    base = f"http://localhost:{API_PORT}"
    try:
        wait_for(f"{base}/health")
        yield base
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def get(url: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


# --- storage ---------------------------------------------------------------------------------


def test_ingest_lands_partitions_in_gcs(pipeline: dict[str, Any]) -> None:
    files = set(pipeline["first"])
    for d in SAMPLE_DAYS:
        for ds in ("system_prices", "mid", "fuelhh", "agile"):
            assert f"raw/{ds}/date={d}/part.parquet" in files
    for d in ("2025-06-09", "2025-06-10"):
        assert f"raw/ndf/date={d}/part.parquet" in files
        assert f"raw/windfor/date={d}/part.parquet" in files
    assert not [f for f in files if ".tmp" in f]


def test_reingest_is_byte_identical(pipeline: dict[str, Any]) -> None:
    assert pipeline["second"] == pipeline["first"]


def test_dst_partition_has_46_periods(bucket: Storage, pipeline: dict[str, Any]) -> None:
    df = bucket.read_parquet("raw/system_prices/date=2025-03-30/part.parquet")
    assert sorted(df["settlement_period"]) == list(range(1, 47))


def test_train_and_plan_write_artefacts(bucket: Storage, pipeline: dict[str, Any]) -> None:
    latest = bucket.read_text("models/lgbm_quantile/LATEST").strip()
    assert latest in pipeline["train"]
    assert bucket.exists(f"{latest}/p50.txt") and bucket.exists(f"{latest}/metadata.json")
    for f in ("forecast.parquet", "schedule.parquet", "plan.json"):
        assert bucket.exists(f"plans/date=2025-06-11/{f}")
    assert json.loads(bucket.read_text("plans/date=2025-06-11/plan.json"))["model"] == latest


def test_failed_upstream_fails_the_job() -> None:
    """No fixtures for this day: the job must exit non-zero, so Cloud Run marks it failed."""
    result = run_job("ingest", "--start", "2024-01-01", "--end", "2024-01-01", "--datasets", "mid")
    assert result.returncode == 1
    assert "failed days (1): 2024-01-01" in result.stderr


# --- API -------------------------------------------------------------------------------------


def test_health_reports_model(api: str) -> None:
    status, body = get(f"{api}/health")
    assert status == 200 and body["latest_model"].startswith("models/lgbm_quantile/")


def test_schedule_contract(api: str) -> None:
    status, body = get(f"{api}/schedule?date=2025-06-11")
    assert status == 200
    assert body["decision_time"] == "2025-06-10T08:00:00Z"
    assert len(body["periods"]) == 48
    assert set(body["periods"][0]) == {"start_time", "charge_kw", "discharge_kw", "soc_kwh"}
    assert all(0 <= p["soc_kwh"] <= 10 + 1e-6 for p in body["periods"])


def test_forecast_contract(api: str) -> None:
    status, body = get(f"{api}/forecast?date=2025-06-11")
    assert status == 200 and len(body["periods"]) == 48
    assert all(p["p10"] <= p["p50"] <= p["p90"] for p in body["periods"])


def test_unplanned_day_is_404(api: str) -> None:
    assert get(f"{api}/schedule?date=2030-01-01")[0] == 404


def test_metrics_score_the_plan_against_actuals(api: str) -> None:
    status, body = get(f"{api}/metrics?end=2025-06-12")
    assert status == 200 and body["days"] == 1 and body["from"] == "2025-06-11"


def test_openapi_docs(api: str) -> None:
    assert get(f"{api}/openapi.json")[0] == 200


# --- image -----------------------------------------------------------------------------------


def test_image_runs_as_non_root() -> None:
    uid = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "id", "flexcast:local", "-u"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert uid == "10001"


def test_image_compressed_size_under_500mb() -> None:
    """Compressed size, measured the same way on every Docker (`docker image inspect .Size`
    reports compressed on containerd image stores such as OrbStack, uncompressed elsewhere)."""
    save = subprocess.Popen(["docker", "save", "flexcast:local"], stdout=subprocess.PIPE)
    gz = subprocess.run(["gzip", "-1"], stdin=save.stdout, capture_output=True, check=True)
    assert save.wait() == 0
    assert len(gz.stdout) < 500 * 1024 * 1024
