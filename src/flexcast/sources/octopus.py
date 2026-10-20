"""Octopus Energy client: Agile import unit rates (known prices for optimiser validation)."""

from datetime import datetime
from typing import Any

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict

from flexcast.sources.http import TIMEOUT, Parsed, get_json, iso_minute, parse_rows


class AgileRate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    valid_from: AwareDatetime
    valid_to: AwareDatetime
    value_exc_vat: float  # p/kWh
    value_inc_vat: float  # p/kWh


class AgileProduct(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    code: str
    available_from: AwareDatetime
    available_to: AwareDatetime | None


class OctopusClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)

    def _paged(self, path: str, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Follow `next` links; Octopus pages default to 100 results."""
        results: list[dict[str, Any]] = []
        url: str | None = f"{self.base_url}{path}"
        while url:
            body = get_json(self.client, url, params)
            results += body["results"]
            url, params = body.get("next"), None  # `next` already carries the query
        return results

    def find_agile_product(self, at: datetime) -> AgileProduct:
        """The domestic Agile import product on sale at `at`. Codes rotate, so never hardcode."""
        products = self._paged("/products/", {"available_at": iso_minute(at)})
        agile = [
            AgileProduct.model_validate(p)
            for p in products
            if p["code"].startswith("AGILE-")
            and p["direction"] == "IMPORT"
            and p["brand"] == "OCTOPUS_ENERGY"  # AGILE-BB-* is the same tariff under the Bulb brand
            and not p["is_business"]
        ]
        if len(agile) != 1:
            raise LookupError(
                f"expected one Agile import product at {at}, got {[a.code for a in agile]}"
            )
        return agile[0]

    def tariff_code(self, product_code: str, region: str) -> str:
        body = get_json(self.client, f"{self.base_url}/products/{product_code}/")
        code: str = body["single_register_electricity_tariffs"][f"_{region}"][
            "direct_debit_monthly"
        ]["code"]
        return code

    def agile_rates(
        self, product_code: str, tariff_code: str, start: datetime, end: datetime
    ) -> Parsed[AgileRate]:
        path = f"/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
        params = {"period_from": iso_minute(start), "period_to": iso_minute(end)}
        return parse_rows(AgileRate, self._paged(path, params))
