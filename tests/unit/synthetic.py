"""Synthetic history for feature/leakage tests. Values encode their origin so tests can check
exactly which row a feature came from:
- price = london_day_ordinal % 97 + 2 * half_hour_of_local_day: unique per (day, clock time)
  within a 97-day window, with a realistic ~£94/MWh daily spread. MID = price + 0.5
- NDF demand / WINDFOR generation = publish time in minutes since 2025-01-01
"""

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from flexcast.history import History
from flexcast.settlement import LONDON, london_day_bounds, period_starts

EPOCH = datetime(2025, 1, 1, tzinfo=UTC)


def encode_price(t: pd.Timestamp) -> float:
    local = t.tz_convert(LONDON)
    half_hour = local.hour * 2 + local.minute // 30
    return float(local.date().toordinal() % 97 + 2 * half_hour)


def encode_publish(t: pd.Timestamp) -> int:
    return int((t - EPOCH).total_seconds() // 60)


def make_history(first: date, last: date) -> History:
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    starts = pd.DatetimeIndex([s for d in days for s in period_starts(d)])
    prices = pd.DataFrame(
        {
            "start_time": starts,
            "created_date_time": starts + pd.Timedelta(days=1),
            "system_sell_price": [encode_price(t) for t in starts],
        }
    )
    apx = pd.DataFrame(
        {
            "start_time": starts,
            "data_provider": "APXMIDP",
            "price": prices["system_sell_price"] + 0.5,
        }
    )
    n2ex = apx.assign(data_provider="N2EXMIDP", price=-999.0)

    lo, hi = london_day_bounds(first)[0], london_day_bounds(last)[1]
    ndf_rows, wind_rows = [], []
    for pub in pd.date_range(lo, hi, freq="30min"):  # NDF: every 30 min, 48 h ahead
        for s in pd.date_range(
            pub + pd.Timedelta("30min"), pub + pd.Timedelta("48h"), freq="30min"
        ):
            ndf_rows.append((pub, s, encode_publish(pub)))
    for pub in pd.date_range(lo, hi, freq="3h"):  # WINDFOR: every 3 h, hourly, 60 h ahead
        for s in pd.date_range(
            pub.floor("h") - pd.Timedelta("24h"), pub + pd.Timedelta("60h"), freq="1h"
        ):
            wind_rows.append((pub, s, encode_publish(pub)))
    ndf = pd.DataFrame(ndf_rows, columns=["publish_time", "start_time", "demand"])
    wind = pd.DataFrame(wind_rows, columns=["publish_time", "start_time", "generation"])
    return History.from_frames(prices, pd.concat([apx, n2ex]), ndf, wind)
