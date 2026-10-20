from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
import respx

from flexcast.config import Settings
from flexcast.settlement import london_day_bounds
from flexcast.sources.elexon import ElexonClient
from flexcast.sources.http import UpstreamError, get_json, iso_minute, windows
from flexcast.sources.octopus import OctopusClient

S = Settings()
NORMAL, SPRING, AUTUMN = date(2025, 6, 11), date(2025, 3, 30), date(2025, 10, 26)


def no_sleep(_: float) -> None:
    pass


# --- retry -----------------------------------------------------------------------------------


@respx.mock
def test_retries_transient_errors_then_succeeds() -> None:
    route = respx.get("https://x.test/a").mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectError("boom"),
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    delays: list[float] = []
    with httpx.Client() as c:
        assert get_json(c, "https://x.test/a", sleep=delays.append) == {"ok": True}
    assert route.call_count == 4
    assert delays == [0.5, 1.0, 2.0]  # exponential backoff, then Retry-After honoured


@respx.mock
def test_gives_up_after_max_attempts() -> None:
    route = respx.get("https://x.test/a").mock(return_value=httpx.Response(500))
    with httpx.Client() as c, pytest.raises(UpstreamError, match="after 3 attempts"):
        get_json(c, "https://x.test/a", attempts=3, sleep=no_sleep)
    assert route.call_count == 3


@respx.mock
def test_does_not_retry_client_errors() -> None:
    route = respx.get("https://x.test/a").mock(return_value=httpx.Response(400))
    with httpx.Client() as c, pytest.raises(httpx.HTTPStatusError):
        get_json(c, "https://x.test/a", sleep=no_sleep)
    assert route.call_count == 1


# --- chunking --------------------------------------------------------------------------------


def test_windows_split_and_cover_range() -> None:
    a = datetime(2025, 6, 1, tzinfo=UTC)
    w = windows(a, a + timedelta(days=2, hours=12), timedelta(days=1))
    assert [(x - a, y - a) for x, y in w] == [
        (timedelta(0), timedelta(days=1)),
        (timedelta(days=1), timedelta(days=2)),
        (timedelta(days=2), timedelta(days=2, hours=12)),
    ]
    assert windows(a, a, timedelta(days=1)) == []


@respx.mock
def test_ndf_chunks_publish_range_into_one_day_requests() -> None:
    route = respx.get(f"{S.elexon_base_url}/datasets/NDF").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    a = datetime(2025, 6, 1, tzinfo=UTC)
    ElexonClient(S.elexon_base_url).ndf(a, a + timedelta(days=3))
    assert route.call_count == 3
    froms = [c.request.url.params["publishDateTimeFrom"] for c in route.calls]
    assert froms == ["2025-06-01T00:00Z", "2025-06-02T00:00Z", "2025-06-03T00:00Z"]


@respx.mock
def test_mid_chunks_into_seven_day_inclusive_ranges() -> None:
    route = respx.get(f"{S.elexon_base_url}/datasets/MID").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    a = datetime(2025, 6, 1, tzinfo=UTC)
    ElexonClient(S.elexon_base_url).mid(a, a + timedelta(days=10))
    params = [dict(c.request.url.params) for c in route.calls]
    assert params == [
        {"from": "2025-06-01T00:00Z", "to": "2025-06-07T23:30Z"},
        {"from": "2025-06-08T00:00Z", "to": "2025-06-10T23:30Z"},
    ]


def test_iso_minute_rejects_naive() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        iso_minute(datetime(2025, 6, 1))  # noqa: DTZ001


# --- parsing recorded fixtures ---------------------------------------------------------------


@pytest.mark.parametrize(("d", "n"), [(NORMAL, 48), (SPRING, 46), (AUTUMN, 50)])
def test_system_prices_parse(upstream: respx.MockRouter, d: date, n: int) -> None:
    p = ElexonClient(S.elexon_base_url).system_prices(d)
    assert not p.rejects
    assert [r.settlement_period for r in p.rows] == list(range(1, n + 1))
    assert all(r.start_time.utcoffset() == timedelta(0) for r in p.rows)


def test_mid_keeps_both_providers(upstream: respx.MockRouter) -> None:
    p = ElexonClient(S.elexon_base_url).mid(*london_day_bounds(AUTUMN))
    providers = {r.data_provider: 0 for r in p.rows}
    for r in p.rows:
        providers[r.data_provider] += 1
    assert providers == {"APXMIDP": 50, "N2EXMIDP": 46}


def test_forecasts_parse(upstream: respx.MockRouter) -> None:
    c = ElexonClient(S.elexon_base_url)
    window = london_day_bounds(date(2025, 6, 10))
    ndf, wind = c.ndf(*window), c.windfor(*window)
    assert len(ndf.rows) == 2816 and not ndf.rejects
    assert len(wind.rows) == 584 and not wind.rejects
    assert len({r.publish_time for r in ndf.rows}) == 48


def test_fuelhh_parse(upstream: respx.MockRouter) -> None:
    p = ElexonClient(S.elexon_base_url).fuelhh(SPRING, SPRING)
    assert len(p.rows) == 46 * 20 and not p.rejects


def test_inconsistent_settlement_row_is_rejected_not_dropped() -> None:
    raw = {
        "settlementDate": "2025-06-11",
        "settlementPeriod": 2,  # 23:00Z is SP1 on this day
        "startTime": "2025-06-10T23:00:00Z",
        "createdDateTime": "2025-06-11T23:44:22Z",
        "systemSellPrice": 1.0,
        "systemBuyPrice": 1.0,
    }
    with respx.mock:
        respx.get(f"{S.elexon_base_url}/balancing/settlement/system-prices/2025-06-11").mock(
            return_value=httpx.Response(200, json={"data": [raw]})
        )
        p = ElexonClient(S.elexon_base_url).system_prices(NORMAL)
    assert p.rows == []
    assert p.rejects[0]["raw"] == raw
    assert "does not match" in p.rejects[0]["error"]


def test_agile_discovery_and_rates(upstream: respx.MockRouter) -> None:
    c = OctopusClient(S.octopus_base_url)
    start, end = london_day_bounds(SPRING)
    product = c.find_agile_product(start)
    assert product.code == "AGILE-24-10-01"
    tariff = c.tariff_code(product.code, "C")
    assert tariff == "E-1R-AGILE-24-10-01-C"
    rates = c.agile_rates(product.code, tariff, start, end)
    assert len(rates.rows) == 46 and not rates.rejects
