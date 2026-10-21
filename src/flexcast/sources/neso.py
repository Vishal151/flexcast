"""NESO data portal client: embedded (distribution-connected) solar and wind forecasts.

CKAN datastore API. The yearly archive resources keep every forecast vintage (Forecast_Datetime);
the live resource only holds the latest snapshot, and the current year's archive is kept up to
date, so we read archives only. Details and quirks: docs/data-sources.md.
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from flexcast.settlement import LONDON, period_starts
from flexcast.sources.http import TIMEOUT, Parsed, get_json, parse_rows, windows

PACKAGE = "embedded-wind-and-solar-forecasts"
ARCHIVE_PREFIX = "Embedded Solar and Wind Forecast Archive "
MAX_HORIZON_DAYS = 2  # D-1 decisions need at most "published on D-2, for D"
COLUMNS = [
    "Forecast_Datetime",
    "SETTLEMENT_DATE",
    "SETTLEMENT_PERIOD",
    "TIME_GMT",
    "EMBEDDED_SOLAR_FORECAST",
    "EMBEDDED_SOLAR_CAPACITY",
    "EMBEDDED_WIND_FORECAST",
    "EMBEDDED_WIND_CAPACITY",
]


def _naive(dt: datetime) -> str:
    """NESO timestamps are naive; we treat them as UTC (see docs/data-sources.md)."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def embedded_sql(resource_id: str, publish_from: datetime, publish_to: datetime) -> str:
    """Vintages published in [publish_from, publish_to), up to MAX_HORIZON_DAYS ahead."""
    last_publish_day = (publish_to - timedelta(microseconds=1)).astimezone(LONDON).date()
    last_target = last_publish_day + timedelta(days=MAX_HORIZON_DAYS)
    cols = ", ".join(f'"{c}"' for c in COLUMNS)
    return (
        f'SELECT {cols} FROM "{resource_id}" '
        f"WHERE \"Forecast_Datetime\" >= '{_naive(publish_from)}' "
        f"AND \"Forecast_Datetime\" < '{_naive(publish_to)}' "
        f"AND \"SETTLEMENT_DATE\" <= '{last_target.isoformat()}T00:00:00' "
        f'ORDER BY "Forecast_Datetime", "SETTLEMENT_DATE", "SETTLEMENT_PERIOD"'
    )


class EmbeddedForecast(BaseModel):
    """One vintage's forecast for one settlement period. start_time is derived from the
    settlement date/period with our helper and cross-checked against NESO's own GMT time."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    publish_time: datetime
    settlement_date: date
    settlement_period: int
    period_end_gmt: time
    solar_mw: float
    solar_capacity_mw: float
    wind_mw: float
    wind_capacity_mw: float
    start_time: datetime | None = None

    @classmethod
    def from_row(cls, r: dict[str, Any]) -> dict[str, Any]:
        return {
            "publish_time": r["Forecast_Datetime"],
            "settlement_date": str(r["SETTLEMENT_DATE"])[:10],
            "settlement_period": r["SETTLEMENT_PERIOD"],
            "period_end_gmt": r["TIME_GMT"],
            "solar_mw": r["EMBEDDED_SOLAR_FORECAST"],
            "solar_capacity_mw": r["EMBEDDED_SOLAR_CAPACITY"],
            "wind_mw": r["EMBEDDED_WIND_FORECAST"],
            "wind_capacity_mw": r["EMBEDDED_WIND_CAPACITY"],
        }

    @field_validator("publish_time")
    @classmethod
    def _as_utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)

    @model_validator(mode="after")
    def _derive_start(self) -> Self:
        starts = period_starts(self.settlement_date)
        if not 1 <= self.settlement_period <= len(starts):
            raise ValueError(
                f"period {self.settlement_period} out of range for {self.settlement_date}"
            )
        start = starts[self.settlement_period - 1]
        if (start + timedelta(minutes=30)).time() != self.period_end_gmt:
            raise ValueError(
                f"{self.settlement_date}/{self.settlement_period} ends at "
                f"{(start + timedelta(minutes=30)).time()} UTC, NESO says {self.period_end_gmt}"
            )
        object.__setattr__(self, "start_time", start)
        return self


class NesoClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
        self._archives: dict[int, str] | None = None
        self._day_ahead: str | None = None

    def archive_resources(self) -> dict[int, str]:
        """Year -> resource id of the yearly archive (discovered, not hardcoded)."""
        if self._archives is None:
            body = get_json(self.client, f"{self.base_url}/package_show", {"id": PACKAGE})
            self._archives = {
                int(r["name"].removeprefix(ARCHIVE_PREFIX)): r["id"]
                for r in body["result"]["resources"]
                if r["name"].startswith(ARCHIVE_PREFIX)
            }
        return self._archives

    def embedded_forecasts(
        self, publish_from: datetime, publish_to: datetime
    ) -> Parsed[EmbeddedForecast]:
        out: Parsed[EmbeddedForecast] = Parsed()
        for a, b in windows(publish_from, publish_to, timedelta(days=1)):
            resource = self.archive_resources()[a.astimezone(UTC).year]
            params = {"sql": embedded_sql(resource, a, b)}
            result = get_json(self.client, f"{self.base_url}/datastore_search_sql", params)
            result = result["result"]
            if result.get("records_truncated"):
                raise RuntimeError(f"NESO truncated the result for {a}..{b}; narrow the window")
            rows = [EmbeddedForecast.from_row(r) for r in result["records"]]
            out.extend(parse_rows(EmbeddedForecast, rows))
        return out

    def day_ahead_prices(self, start: datetime, end: datetime) -> "Parsed[DayAheadPrice]":
        return day_ahead_prices(self, start, end)


# --- N2EX GB day-ahead auction prices --------------------------------------------------------

DAY_AHEAD_PACKAGE = "day-ahead-power-exchange-prices-nordpool"


def day_ahead_sql(resource_id: str, first_utc_date: date, last_utc_date: date) -> str:
    return (
        f'SELECT "Date", "Delivery Period", "Price" FROM "{resource_id}" '
        f"WHERE \"Date\" >= '{first_utc_date.isoformat()}' "
        f"AND \"Date\" <= '{last_utc_date.isoformat()}' "
        f'ORDER BY "Date", "Delivery Period"'
    )


class DayAheadPrice(BaseModel):
    """One hourly auction price. NESO's Date/Delivery Period are UTC (verified against Octopus
    Agile; docs/data-sources.md), so start_time = Date + hour, in UTC."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    start_time: datetime
    price: float

    @classmethod
    def from_row(cls, r: dict[str, Any]) -> dict[str, Any]:
        period = str(r["Delivery Period"])
        start_h, end_h = (int(part.strip()[:2]) for part in period.split("-"))
        if (start_h + 1) % 24 != end_h:
            raise ValueError(f"unexpected delivery period {period!r}")
        day = date.fromisoformat(str(r["Date"])[:10])
        return {
            "start_time": datetime.combine(day, time(start_h), UTC),
            "price": r["Price"],
        }


def _day_ahead_resource(client: "NesoClient") -> str:
    """Discovered once per client (the portal could re-issue the resource id)."""
    if client._day_ahead is None:
        body = get_json(client.client, f"{client.base_url}/package_show", {"id": DAY_AHEAD_PACKAGE})
        (resource,) = [r for r in body["result"]["resources"] if "N2EX" in r["name"]]
        client._day_ahead = str(resource["id"])
    return client._day_ahead


def day_ahead_prices(client: "NesoClient", start: datetime, end: datetime) -> Parsed[DayAheadPrice]:
    """Hourly prices with start_time in [start, end) (UTC hours; a London day has 23-25)."""
    rid = _day_ahead_resource(client)
    first = start.astimezone(UTC).date()
    last = (end - timedelta(seconds=1)).astimezone(UTC).date()
    params = {"sql": day_ahead_sql(rid, first, last)}
    result = get_json(client.client, f"{client.base_url}/datastore_search_sql", params)["result"]
    out: Parsed[DayAheadPrice] = Parsed()
    for r in result["records"]:
        try:
            row = DayAheadPrice.from_row(r)
        except (ValueError, KeyError) as e:
            out.rejects.append({"error": str(e), "raw": r})
            continue
        if start <= row["start_time"] < end:
            out.extend(parse_rows(DayAheadPrice, [row]))
    return out
