import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from flexcast.config import Settings
from flexcast.ingest import DATASETS, Ingester, merge
from flexcast.settlement import LONDON
from flexcast.sources.elexon import ElexonClient
from flexcast.sources.gas import GasClient
from flexcast.sources.neso import NesoClient
from flexcast.sources.octopus import OctopusClient
from flexcast.storage import Storage

S = Settings()
NORMAL, SPRING, AUTUMN = date(2025, 6, 11), date(2025, 3, 30), date(2025, 10, 26)
DAYS = {NORMAL: 48, SPRING: 46, AUTUMN: 50}
# The fixtures hold publish windows for D-1 and D-2 of each sample day.
FORECAST_DAYS = [date(2025, 6, 10), date(2025, 6, 9), date(2025, 3, 29), date(2025, 10, 25)]
EMBEDDED_DAYS = [date(2025, 6, 9), date(2025, 6, 10), date(2025, 10, 25)]


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(f"file://{tmp_path}")


@pytest.fixture
def ingester(storage: Storage) -> Ingester:
    return Ingester(
        storage,
        ElexonClient(S.elexon_base_url),
        OctopusClient(S.octopus_base_url),
        region="C",
        neso=NesoClient(S.neso_base_url),
        gas=GasClient(S.gas_base_url),
    )


def ingest_all_fixtures(ing: Ingester) -> None:
    for d in DAYS:
        ing.ingest_day(d, ["system_prices", "mid", "fuelhh", "agile", "day_ahead"])
    for d in FORECAST_DAYS:
        ing.ingest_day(d, ["ndf", "windfor"])
    for d in EMBEDDED_DAYS:
        ing.ingest_day(d, ["embedded"])


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_rerun_produces_identical_files(
    upstream: respx.MockRouter, ingester: Ingester, tmp_path: Path
) -> None:
    ingest_all_fixtures(ingester)
    first = tree_hashes(tmp_path)
    ingest_all_fixtures(ingester)
    assert tree_hashes(tmp_path) == first
    assert len(first) == 5 * len(DAYS) + 2 * len(FORECAST_DAYS) + len(EMBEDDED_DAYS)


@pytest.mark.parametrize(("d", "n"), DAYS.items())
def test_dst_days_have_right_period_count(
    upstream: respx.MockRouter, ingester: Ingester, storage: Storage, d: date, n: int
) -> None:
    ingester.ingest_day(d, ["system_prices", "agile"])
    prices = storage.read_parquet(f"raw/system_prices/date={d}/part.parquet")
    assert prices["settlement_period"].tolist() == list(range(1, n + 1))
    assert str(prices["start_time"].dt.tz) == "UTC"
    agile = storage.read_parquet(f"raw/agile/date={d}/part.parquet")
    assert len(agile) == n
    assert set(agile["product_code"]) == {"AGILE-24-10-01"} and set(agile["region"]) == {"C"}


def test_forecast_partition_holds_only_that_london_days_publishes(
    upstream: respx.MockRouter, ingester: Ingester, storage: Storage
) -> None:
    ingester.ingest_day(date(2025, 6, 10), ["ndf"])
    ndf = storage.read_parquet("raw/ndf/date=2025-06-10/part.parquet")
    assert {t.astimezone(LONDON).date() for t in ndf["publish_time"]} == {date(2025, 6, 10)}
    assert not ndf.duplicated(list(DATASETS["ndf"].key)).any()


def test_merge_dedupes_on_key_keeps_revisions_and_sorts() -> None:
    t = pd.Timestamp("2025-06-10T23:00Z")
    existing = pd.DataFrame(
        {"start_time": [t, t], "created": [1, 2], "price": [10.0, 11.0]}
    )  # two revisions of one period
    new = pd.DataFrame(
        {"start_time": [t, t - pd.Timedelta("30min")], "created": [2, 1], "price": [11.0, 9.0]}
    )
    out = merge(existing, new, ("start_time", "created"))
    assert out.to_dict("list") == {
        "start_time": [t - pd.Timedelta("30min"), t, t],
        "created": [1, 1, 2],
        "price": [9.0, 10.0, 11.0],
    }


def test_bad_rows_go_to_rejects_and_good_rows_are_kept(
    ingester: Ingester, storage: Storage
) -> None:
    good = {
        "settlementDate": "2025-06-11",
        "settlementPeriod": 1,
        "startTime": "2025-06-10T23:00:00Z",
        "createdDateTime": "2025-06-11T23:44:22Z",
        "systemSellPrice": 50.0,
        "systemBuyPrice": 50.0,
    }
    bad = {**good, "settlementPeriod": 2, "systemSellPrice": "n/a"}
    with respx.mock:
        respx.get(f"{S.elexon_base_url}/balancing/settlement/system-prices/2025-06-11").mock(
            return_value=httpx.Response(200, json={"data": [good, bad]})
        )
        result = ingester.ingest_day(NORMAL, ["system_prices"])
    assert result["system_prices"] == (1, 1)
    assert len(storage.read_parquet("raw/system_prices/date=2025-06-11/part.parquet")) == 1
    (line,) = storage.read_text("rejects/system_prices/date=2025-06-11/rejects.jsonl").splitlines()
    assert json.loads(line)["raw"] == bad


def test_one_dataset_failing_does_not_stop_the_others(ingester: Ingester, storage: Storage) -> None:
    fixture = Path(__file__).parents[1] / "fixtures/elexon/system_prices_normal.json"
    with respx.mock, pytest.raises(ExceptionGroup) as exc:
        respx.get(f"{S.elexon_base_url}/datasets/MID").mock(return_value=httpx.Response(400))
        respx.get(f"{S.elexon_base_url}/balancing/settlement/system-prices/2025-06-11").mock(
            return_value=httpx.Response(200, content=fixture.read_bytes())
        )
        ingester.ingest_day(NORMAL, ["mid", "system_prices"])
    assert len(exc.value.exceptions) == 1
    assert storage.exists("raw/system_prices/date=2025-06-11/part.parquet")


def test_empty_response_writes_nothing(ingester: Ingester, storage: Storage) -> None:
    with respx.mock:
        respx.get(f"{S.elexon_base_url}/balancing/settlement/system-prices/2025-06-11").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        ingester.ingest_day(NORMAL, ["system_prices"])
    assert storage.list("raw") == []


def test_default_range_is_yesterday_and_today() -> None:
    from flexcast.ingest import default_range

    now = datetime(2025, 6, 10, 23, 30, tzinfo=UTC)  # already 11 June in London
    assert default_range(now) == (date(2025, 6, 10), date(2025, 6, 11))


def test_embedded_partition_has_derived_utc_start_times(
    upstream: respx.MockRouter, ingester: Ingester, storage: Storage
) -> None:
    ingester.ingest_day(date(2025, 6, 10), ["embedded"])
    df = storage.read_parquet("raw/embedded/date=2025-06-10/part.parquet")
    assert {"publish_time", "start_time", "solar_mw", "solar_capacity_mw", "wind_mw"} <= set(df)
    assert "period_end_gmt" not in df
    assert str(df["start_time"].dt.tz) == "UTC" and str(df["publish_time"].dt.tz) == "UTC"
    assert not df.duplicated(["publish_time", "start_time"]).any()


@pytest.mark.parametrize(("d", "hours"), [(NORMAL, 24), (SPRING, 23), (AUTUMN, 25)])
def test_day_ahead_partition_is_one_london_day(
    upstream: respx.MockRouter, ingester: Ingester, storage: Storage, d: date, hours: int
) -> None:
    ingester.ingest_day(d, ["day_ahead"])
    df = storage.read_parquet(f"raw/day_ahead/date={d}/part.parquet")
    assert len(df) == hours and str(df["start_time"].dt.tz) == "UTC"


def test_candidate_datasets_ingest_idempotently(
    upstream: respx.MockRouter, ingester: Ingester, storage: Storage, tmp_path: Path
) -> None:
    pub_day, gas_day = date(2025, 6, 9), date(2025, 6, 5)

    def ingest() -> None:
        ingester.ingest_day(pub_day, ["availability"])
        ingester.ingest_day(gas_day, ["gas_sap"])

    ingest()
    first = tree_hashes(tmp_path)
    ingest()
    assert tree_hashes(tmp_path) == first
    av = storage.read_parquet(f"raw/availability/date={pub_day}/part.parquet")
    assert av["publish_time"].dt.tz_convert(LONDON).dt.date.eq(pub_day).all()
    assert av["publish_time"].nunique() == 24  # hourly vintages
    gas = storage.read_parquet(f"raw/gas_sap/date={gas_day}/part.parquet")
    assert gas["gas_day"].tolist() == [gas_day]
    assert 1 < gas["price_p_per_kwh"].iloc[0] < 20
