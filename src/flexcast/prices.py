"""Load one settlement day of prices from storage as a £/MWh series indexed by UTC period start."""

from datetime import date
from typing import Literal

import pandas as pd

from flexcast.settlement import period_starts
from flexcast.storage import Storage

PriceSource = Literal["system", "agile"]


def load_day_prices(storage: Storage, source: PriceSource, d: date) -> pd.Series:
    """Complete day of prices, or raise: a partial day would silently skew the optimiser.

    system: system sell price, latest settlement revision per period.
    agile:  Octopus Agile import rate inc VAT, p/kWh * 10 -> £/MWh.
    """
    if source == "system":
        path = f"raw/system_prices/date={d}/part.parquet"
        if not storage.exists(path):
            raise FileNotFoundError(path)
        df = storage.read_parquet(path).sort_values("created_date_time")
        df = df.drop_duplicates("start_time", keep="last")
        series = df.set_index("start_time")["system_sell_price"]
    else:
        path = f"raw/agile/date={d}/part.parquet"
        if not storage.exists(path):
            raise FileNotFoundError(path)
        df = storage.read_parquet(path)
        series = df.set_index("valid_from")["value_inc_vat"] * 10

    expected = pd.DatetimeIndex(period_starts(d))
    series = series.sort_index()
    if len(series) != len(expected) or not series.index.equals(expected):
        raise ValueError(f"{source} prices for {d}: {len(series)} of {len(expected)} periods")
    return series.rename("price_gbp_per_mwh").rename_axis("start_time")
