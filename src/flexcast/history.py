"""In-memory history with availability times. The one gate between stored data and features.

Every frame carries `available_at`: the earliest instant a real system could have known the row.
Feature code may only read through `as_of(name, decision_time)`, which keeps rows with
available_at < decision_time. Availability rules (docs/data-sources.md, SPEC §2):

- ndf, windfor: publish_time.
- prices (system sell price), mid (APXMIDP): period start + PRICE_LAG. Real-time publication is
  ~48 min after period start; the stored value is the latest revision, which we accept as a
  known limitation (revisions are usually small; ingest now keeps every revision it sees).
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd

from flexcast.settlement import LONDON, london_day_bounds
from flexcast.storage import Storage

PRICE_LAG = timedelta(hours=1)
MID_PROVIDER = "APXMIDP"  # N2EX has gaps and zero-volume periods

Name = Literal["prices", "mid", "ndf", "windfor"]


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
    _ndf_by_day: dict[date, pd.DataFrame] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Index NDF (2M rows) by target London date once, so each feature build is cheap.
        local_date = self.ndf["start_time"].dt.tz_convert(LONDON).dt.date
        object.__setattr__(
            self, "_ndf_by_day", {d: g for d, g in self.ndf.groupby(local_date, sort=False)}
        )

    @classmethod
    def from_frames(
        cls,
        prices: pd.DataFrame,
        mid: pd.DataFrame,
        ndf: pd.DataFrame,
        windfor: pd.DataFrame,
    ) -> "History":
        """Build from raw-schema frames (as stored by ingest)."""
        p = prices.sort_values("created_date_time").drop_duplicates("start_time", keep="last")
        p = p.rename(columns={"system_sell_price": "price"})[["start_time", "price"]]
        p = _with_london_keys(p.assign(available_at=p["start_time"] + PRICE_LAG))

        m = mid[mid["data_provider"] == MID_PROVIDER][["start_time", "price"]]
        m = _with_london_keys(m.assign(available_at=m["start_time"] + PRICE_LAG))

        n = ndf[["publish_time", "start_time", "demand"]]
        w = windfor[["publish_time", "start_time", "generation"]]
        return cls(
            prices=p.sort_values("start_time").reset_index(drop=True),
            mid=m.sort_values("start_time").reset_index(drop=True),
            ndf=n.assign(available_at=n["publish_time"]).reset_index(drop=True),
            windfor=w.assign(available_at=w["publish_time"]).reset_index(drop=True),
        )

    @classmethod
    def load(cls, storage: Storage) -> "History":
        return cls.from_frames(
            storage.read_parquet_dir("raw/system_prices"),
            storage.read_parquet_dir("raw/mid"),
            storage.read_parquet_dir("raw/ndf"),
            storage.read_parquet_dir("raw/windfor"),
        )

    def as_of(
        self, name: Name, decision_time: datetime, target_date: date | None = None
    ) -> pd.DataFrame:
        """Rows knowable strictly before decision_time. For ndf, target_date narrows to that
        day's periods first (a speed-up; the availability filter is the same)."""
        if decision_time.tzinfo is None:
            raise ValueError("decision_time must be tz-aware")
        if name == "ndf" and target_date is not None:
            df = self._ndf_by_day.get(target_date, self.ndf.iloc[:0])
        else:
            df = getattr(self, name)
        out: pd.DataFrame = df[df["available_at"] < decision_time]
        return out

    def actual_prices(self, target_date: date) -> pd.Series:
        """Outturn prices for a day: for labels and settlement only, NEVER for features."""
        start, end = london_day_bounds(target_date)
        p = self.prices
        day = p[(p["start_time"] >= start) & (p["start_time"] < end)]
        return day.set_index("start_time")["price"].rename("price_gbp_per_mwh")
