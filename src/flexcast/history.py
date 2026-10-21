"""In-memory history with availability times. The one gate between stored data and features.

Every frame carries `available_at`: the earliest instant a real system could have known the row.
Feature code may only read through `as_of(name, decision_time)`, which keeps rows with
available_at < decision_time. Availability rules (docs/data-sources.md, SPEC §2):

- ndf, windfor: publish_time.
- day_ahead (N2EX auction prices): 13:00 London on D-1 for London day D. The auction publishes by
  10:00 D-1; NESO republishes ~12:30 (measured), plus a half-hour margin (docs/data-sources.md).
- embedded (NESO solar/wind): publish_time + EMBEDDED_LAG. A vintage reaches the archive some
  25-85 min after its timestamp, and the timestamp's timezone is unconfirmed; 2 h covers both.
- prices (system sell price), mid (APXMIDP): period start + PRICE_LAG. Real-time publication is
  ~48 min after period start; the stored value is the latest revision, which we accept as a
  known limitation (revisions are usually small; ingest now keeps every revision it sees).
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Literal

import pandas as pd

from flexcast.settlement import LONDON, london_day_bounds
from flexcast.storage import Storage

PRICE_LAG = timedelta(hours=1)
EMBEDDED_LAG = timedelta(hours=2)
DAY_AHEAD_AVAILABLE = time(13)  # London wall-clock on D-1; NESO republishes ~12:30 (observed)
MID_PROVIDER = "APXMIDP"  # N2EX has gaps and zero-volume periods

Name = Literal["prices", "mid", "ndf", "windfor", "embedded", "day_ahead"]
Market = Literal["imbalance", "day_ahead"]


UTC_TIME = "datetime64[us, UTC]"
# The raw columns from_frames needs, so an empty bucket loads as an empty history.
EMPTY_SCHEMAS: dict[str, dict[str, str]] = {
    "system_prices": {
        "start_time": UTC_TIME,
        "created_date_time": UTC_TIME,
        "system_sell_price": "float64",
    },
    "mid": {"start_time": UTC_TIME, "data_provider": "object", "price": "float64"},
    "ndf": {"publish_time": UTC_TIME, "start_time": UTC_TIME, "demand": "int64"},
    "windfor": {"publish_time": UTC_TIME, "start_time": UTC_TIME, "generation": "int64"},
    "day_ahead": {"start_time": UTC_TIME, "price": "float64"},
    "embedded": {
        "publish_time": UTC_TIME,
        "start_time": UTC_TIME,
        "solar_mw": "float64",
        "solar_capacity_mw": "float64",
        "wind_mw": "float64",
    },
}


def _empty(schema: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})


def _day_ahead_half_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Each hourly auction price applies to both half-hours of its hour; available from
    DAY_AHEAD_AVAILABLE London on the day before its London date."""
    h = hourly[["start_time", "price"]]
    halves = pd.concat([h, h.assign(start_time=h["start_time"] + pd.Timedelta(minutes=30))])
    halves = _with_london_keys(halves.sort_values("start_time").reset_index(drop=True))
    published = [
        datetime.combine(d - timedelta(days=1), DAY_AHEAD_AVAILABLE, LONDON)
        for d in halves["london_date"]
    ]
    return halves.assign(available_at=pd.to_datetime(published, utc=True))


def _with_london_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add London date and clock time (HH:MM) for 'same time on another day' lookups."""
    local = df["start_time"].dt.tz_convert(LONDON)
    return df.assign(london_date=local.dt.date, hhmm=local.dt.strftime("%H:%M"))


@dataclass(frozen=True)
class History:
    prices: pd.DataFrame  # start_time, price, available_at, london_date, hhmm
    mid: pd.DataFrame  # start_time, price, available_at, london_date, hhmm
    ndf: pd.DataFrame  # publish_time, start_time, demand, available_at
    windfor: pd.DataFrame  # publish_time, start_time (hourly), generation, available_at
    embedded: pd.DataFrame  # publish_time, start_time, solar_mw, solar_capacity_mw, wind_mw, ...
    day_ahead: pd.DataFrame  # start_time (half-hourly), price, available_at, london_date, hhmm
    _by_day: dict[str, dict[date, pd.DataFrame]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Index the big vintage tables (millions of rows) by target London date once, so each
        # feature build only scans one day's rows.
        by_day = {}
        for name in ("ndf", "embedded"):
            df: pd.DataFrame = getattr(self, name)
            local_date = df["start_time"].dt.tz_convert(LONDON).dt.date
            by_day[name] = {d: g for d, g in df.groupby(local_date, sort=False)}
        object.__setattr__(self, "_by_day", by_day)

    @classmethod
    def from_frames(
        cls,
        prices: pd.DataFrame,
        mid: pd.DataFrame,
        ndf: pd.DataFrame,
        windfor: pd.DataFrame,
        embedded: pd.DataFrame | None = None,
        day_ahead: pd.DataFrame | None = None,
    ) -> "History":
        """Build from raw-schema frames (as stored by ingest)."""
        p = prices.sort_values("created_date_time").drop_duplicates("start_time", keep="last")
        p = p.rename(columns={"system_sell_price": "price"})[["start_time", "price"]]
        p = _with_london_keys(p.assign(available_at=p["start_time"] + PRICE_LAG))

        m = mid[mid["data_provider"] == MID_PROVIDER][["start_time", "price"]]
        m = _with_london_keys(m.assign(available_at=m["start_time"] + PRICE_LAG))

        n = ndf[["publish_time", "start_time", "demand"]]
        w = windfor[["publish_time", "start_time", "generation"]]
        e = _empty(EMPTY_SCHEMAS["embedded"]) if embedded is None else embedded
        e = e[["publish_time", "start_time", "solar_mw", "solar_capacity_mw", "wind_mw"]]
        da = _day_ahead_half_hourly(
            _empty(EMPTY_SCHEMAS["day_ahead"]) if day_ahead is None else day_ahead
        )
        return cls(
            prices=p.sort_values("start_time").reset_index(drop=True),
            mid=m.sort_values("start_time").reset_index(drop=True),
            ndf=n.assign(available_at=n["publish_time"]).reset_index(drop=True),
            windfor=w.assign(available_at=w["publish_time"]).reset_index(drop=True),
            embedded=e.assign(available_at=e["publish_time"] + EMBEDDED_LAG).reset_index(drop=True),
            day_ahead=da,
        )

    @classmethod
    def load(cls, storage: Storage) -> "History":
        def read(dataset: str) -> pd.DataFrame:
            df = storage.read_parquet_dir(f"raw/{dataset}")
            return df if not df.empty else _empty(EMPTY_SCHEMAS[dataset])  # fresh deployment

        return cls.from_frames(
            read("system_prices"),
            read("mid"),
            read("ndf"),
            read("windfor"),
            read("embedded"),
            read("day_ahead"),
        )

    def as_of(
        self, name: Name, decision_time: datetime, target_date: date | None = None
    ) -> pd.DataFrame:
        """Rows knowable strictly before decision_time. For the vintage tables (ndf, embedded),
        target_date narrows to that day's periods first (a speed-up; same availability filter)."""
        if decision_time.tzinfo is None:
            raise ValueError("decision_time must be tz-aware")
        if name in self._by_day and target_date is not None:
            whole: pd.DataFrame = getattr(self, name)
            df = self._by_day[name].get(target_date, whole.iloc[:0])
        else:
            df = getattr(self, name)
        out: pd.DataFrame = df[df["available_at"] < decision_time]
        return out

    def actual_prices(self, target_date: date, market: Market = "imbalance") -> pd.Series:
        """Outturn prices for a day (half-hourly): for labels and settlement only, NEVER for
        features. market="day_ahead" gives the auction price of each half-hour's hour."""
        start, end = london_day_bounds(target_date)
        p = self.prices if market == "imbalance" else self.day_ahead
        day = p[(p["start_time"] >= start) & (p["start_time"] < end)]
        return day.set_index("start_time")["price"].rename("price_gbp_per_mwh")
