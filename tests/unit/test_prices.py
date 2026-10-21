from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import respx

from flexcast.config import Settings
from flexcast.ingest import Ingester
from flexcast.prices import PriceSource, load_day_prices
from flexcast.settlement import period_starts
from flexcast.sources.elexon import ElexonClient
from flexcast.sources.neso import NesoClient
from flexcast.sources.octopus import OctopusClient
from flexcast.storage import Storage

S = Settings()
SPRING = date(2025, 3, 30)


@pytest.fixture
def storage(tmp_path: Path, upstream: respx.MockRouter) -> Storage:
    st = Storage(f"file://{tmp_path}")
    ing = Ingester(
        st,
        ElexonClient(S.elexon_base_url),
        OctopusClient(S.octopus_base_url),
        "C",
        neso=NesoClient(S.neso_base_url),
    )
    ing.ingest_day(SPRING, ["system_prices", "agile"])
    return st


@pytest.mark.parametrize("source", ["system", "agile"])
def test_full_dst_day_indexed_by_period_start(storage: Storage, source: PriceSource) -> None:
    p = load_day_prices(storage, source, SPRING)
    assert list(p.index) == period_starts(SPRING)
    assert p.name == "price_gbp_per_mwh"


def test_agile_converted_from_p_per_kwh_inc_vat(storage: Storage) -> None:
    raw = storage.read_parquet(f"raw/agile/date={SPRING}/part.parquet")
    first = raw.sort_values("valid_from").iloc[0]
    assert load_day_prices(storage, "agile", SPRING).iloc[0] == pytest.approx(
        first["value_inc_vat"] * 10
    )


def test_system_price_uses_latest_revision(storage: Storage) -> None:
    path = f"raw/system_prices/date={SPRING}/part.parquet"
    df = storage.read_parquet(path)
    revised = df.iloc[[0]].copy()
    revised["created_date_time"] += pd.Timedelta(days=30)
    revised["system_sell_price"] = 999.0
    storage.write_parquet(pd.concat([df, revised]), path)
    p = load_day_prices(storage, "system", SPRING)
    assert len(p) == 46 and p.iloc[0] == 999.0


def test_incomplete_day_raises(storage: Storage) -> None:
    path = f"raw/agile/date={SPRING}/part.parquet"
    storage.write_parquet(storage.read_parquet(path).iloc[:40], path)
    with pytest.raises(ValueError, match="40 of 46"):
        load_day_prices(storage, "agile", SPRING)


def test_missing_day_raises(storage: Storage) -> None:
    with pytest.raises(FileNotFoundError):
        load_day_prices(storage, "agile", date(2025, 1, 1))
