"""Elexon Insights (BMRS) client. Endpoints, limits and quirks: docs/data-sources.md."""

from datetime import date, datetime, timedelta
from typing import Any, Self

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator
from pydantic.alias_generators import to_camel

from flexcast.settlement import to_settlement
from flexcast.sources.http import TIMEOUT, Parsed, get_json, iso_minute, parse_rows, windows

PUBLISH_WINDOW_MAX = timedelta(days=1)  # NDF caps publish windows at 1 day
SETTLEMENT_RANGE_MAX = timedelta(days=7)  # MID and FUELHH cap ranges at 7 days


class Row(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, frozen=True, extra="ignore")


class SettlementRow(Row):
    """A row carrying both start_time and settlement date/period; they must agree."""

    start_time: AwareDatetime
    settlement_date: date
    settlement_period: int

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        expected = to_settlement(self.start_time)
        if (self.settlement_date, self.settlement_period) != expected:
            raise ValueError(
                f"settlement {self.settlement_date}/{self.settlement_period} does not match "
                f"start_time {self.start_time.isoformat()} (expected {expected[0]}/{expected[1]})"
            )
        return self


class SystemPrice(SettlementRow):
    created_date_time: AwareDatetime
    system_sell_price: float
    system_buy_price: float
    net_imbalance_volume: float | None = None
    price_derivation_code: str | None = None


class Mid(SettlementRow):
    data_provider: str
    price: float
    volume: float


class Ndf(SettlementRow):
    publish_time: AwareDatetime
    demand: int
    boundary: str


class Windfor(Row):
    publish_time: AwareDatetime
    start_time: AwareDatetime
    generation: int


class Fuelhh(SettlementRow):
    publish_time: AwareDatetime
    fuel_type: str
    generation: int


class ElexonClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)

    def _data(self, path: str, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        body = get_json(self.client, f"{self.base_url}{path}", params)
        rows: list[dict[str, Any]] = body["data"]
        return rows

    def system_prices(self, settlement_date: date) -> Parsed[SystemPrice]:
        path = f"/balancing/settlement/system-prices/{settlement_date.isoformat()}"
        return parse_rows(SystemPrice, self._data(path))

    def mid(self, start: datetime, end: datetime) -> Parsed[Mid]:
        """Both providers, half-hours starting in [start, end). from/to are inclusive instants."""
        out: Parsed[Mid] = Parsed()
        for a, b in windows(start, end, SETTLEMENT_RANGE_MAX):
            params = {"from": iso_minute(a), "to": iso_minute(b - timedelta(minutes=30))}
            out.extend(parse_rows(Mid, self._data("/datasets/MID", params)))
        return out

    def fuelhh(self, first: date, last: date) -> Parsed[Fuelhh]:
        """Settlement dates first..last inclusive."""
        out: Parsed[Fuelhh] = Parsed()
        d = first
        while d <= last:
            stop = min(d + SETTLEMENT_RANGE_MAX - timedelta(days=1), last)
            params = {"settlementDateFrom": d.isoformat(), "settlementDateTo": stop.isoformat()}
            out.extend(parse_rows(Fuelhh, self._data("/datasets/FUELHH", params)))
            d = stop + timedelta(days=1)
        return out

    def _by_publish_time[T: Row](
        self, path: str, model: type[T], start: datetime, end: datetime
    ) -> Parsed[T]:
        out: Parsed[T] = Parsed()
        for a, b in windows(start, end, PUBLISH_WINDOW_MAX):
            params = {"publishDateTimeFrom": iso_minute(a), "publishDateTimeTo": iso_minute(b)}
            out.extend(parse_rows(model, self._data(path, params)))
        return out

    def ndf(self, publish_from: datetime, publish_to: datetime) -> Parsed[Ndf]:
        return self._by_publish_time("/datasets/NDF", Ndf, publish_from, publish_to)

    def windfor(self, publish_from: datetime, publish_to: datetime) -> Parsed[Windfor]:
        return self._by_publish_time("/datasets/WINDFOR", Windfor, publish_from, publish_to)
