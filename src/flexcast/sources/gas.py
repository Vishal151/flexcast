"""National Gas Transmission data portal: the daily System Average Price (SAP) of gas.

SAP is the average price of gas traded on the on-the-day market for one gas day (05:00-05:00 UK).
The portal republishes a rolling window daily at ~11:40 UTC; gas day d first appears on d+1.
No key or registration. The download endpoint only serves CSV. Quirks: docs/data-sources.md.
"""

import csv
import io
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from flexcast.sources.http import TIMEOUT, Parsed, get_text, parse_rows

SAP_ID = "PUBOB603"  # "SAP, Actual Day", p/kWh
SAP_ITEM = "SAP, Actual Day"


class GasPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    gas_day: date
    price_p_per_kwh: float

    @field_validator("gas_day", mode="before")
    @classmethod
    def _uk_date(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        day, month, year = v.split("/")  # DD/MM/YYYY
        return date(int(year), int(month), int(day))


def sap_params(first: date, last: date) -> dict[str, str]:
    return {
        "applicableFor": "Y",
        "dateFrom": first.isoformat(),
        "dateTo": last.isoformat(),
        "dateType": "GASDAY",
        "latestFlag": "Y",
        "ids": SAP_ID,
        "type": "CSV",
    }


class GasClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)

    def sap(self, first: date, last: date) -> Parsed[GasPrice]:
        """Latest SAP for gas days first..last inclusive (an empty body if none published)."""
        url = f"{self.base_url}/find-gas-data-download"
        body = get_text(self.client, url, sap_params(first, last))
        rows = list(csv.DictReader(io.StringIO(body)))
        good = [r for r in rows if r.get("Data Item") == SAP_ITEM]
        out = parse_rows(
            GasPrice,
            ({"gas_day": r["Applicable For"], "price_p_per_kwh": r["Value"]} for r in good),
        )
        out.rejects += [{"error": "unexpected data item", "raw": r} for r in rows if r not in good]
        return out
