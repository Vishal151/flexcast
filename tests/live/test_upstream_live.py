"""Live smoke test against the real Elexon and Octopus APIs (opt-in: `make live`, weekly in CI).

Catches upstream drift: endpoints moving, fields renamed, validation starting to reject rows.
Uses the real clients, so it checks exactly what ingest depends on, for a recent complete day.
"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from flexcast.config import Settings
from flexcast.settlement import LONDON, london_day_bounds, period_starts
from flexcast.sources.elexon import ElexonClient
from flexcast.sources.neso import NesoClient
from flexcast.sources.octopus import OctopusClient

pytestmark = pytest.mark.live

S = Settings()
DAY = datetime.now(UTC).astimezone(LONDON).date() - timedelta(days=3)  # fully published by now
OPENAPI = "https://data.elexon.co.uk/swagger/v1/swagger.json"
PATHS_WE_USE = [
    "/balancing/settlement/system-prices/{settlementDate}",
    "/datasets/MID",
    "/datasets/NDF",
    "/datasets/WINDFOR",
    "/datasets/FUELHH",
]


@pytest.fixture(scope="module")
def elexon() -> ElexonClient:
    return ElexonClient(S.elexon_base_url)


def test_openapi_still_lists_our_endpoints() -> None:
    paths = httpx.get(OPENAPI, timeout=60).raise_for_status().json()["paths"]
    assert [p for p in PATHS_WE_USE if p not in paths] == []


def test_system_prices(elexon: ElexonClient) -> None:
    p = elexon.system_prices(DAY)
    assert not p.rejects, p.rejects[:2]
    assert len(p.rows) == len(period_starts(DAY))


def test_mid_has_apx(elexon: ElexonClient) -> None:
    p = elexon.mid(*london_day_bounds(DAY))
    assert not p.rejects, p.rejects[:2]
    apx = [r for r in p.rows if r.data_provider == "APXMIDP"]
    assert len(apx) == len(period_starts(DAY))


@pytest.mark.parametrize("dataset", ["ndf", "windfor"])
def test_forecast_vintages(elexon: ElexonClient, dataset: str) -> None:
    p = getattr(elexon, dataset)(*london_day_bounds(DAY))
    assert not p.rejects, p.rejects[:2]
    vintages = {r.publish_time for r in p.rows}
    assert len(vintages) >= (40 if dataset == "ndf" else 6)  # ~48/day and 8/day observed


def test_fuelhh(elexon: ElexonClient) -> None:
    p = elexon.fuelhh(DAY, DAY)
    assert not p.rejects, p.rejects[:2]
    assert {r.fuel_type for r in p.rows} >= {"CCGT", "WIND", "NUCLEAR"}


def test_agile_discovery_and_rates() -> None:
    c = OctopusClient(S.octopus_base_url)
    start, end = london_day_bounds(DAY)
    product = c.find_agile_product(start)
    rates = c.agile_rates(product.code, c.tariff_code(product.code, S.agile_region), start, end)
    assert not rates.rejects
    assert len(rates.rows) == len(period_starts(DAY))


def test_neso_embedded_forecast_vintages() -> None:
    p = NesoClient(S.neso_base_url).embedded_forecasts(*london_day_bounds(DAY))
    assert not p.rejects, p.rejects[:2]
    assert len({r.publish_time for r in p.rows}) >= 20  # hourly vintages
    tomorrow = DAY + timedelta(days=1)
    latest = max(r.publish_time for r in p.rows)
    covered = {
        r.settlement_period
        for r in p.rows
        if r.publish_time == latest and r.settlement_date == tomorrow
    }
    assert len(covered) == len(period_starts(tomorrow))
