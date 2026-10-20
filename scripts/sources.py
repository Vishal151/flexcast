"""The upstream requests Stage 0 explores and records. Shared by explore_api.py and
record_fixtures.py so the two cannot drift. Throwaway: Stage 1 replaces this with real clients."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")

BASE_URLS = {
    "elexon": "https://data.elexon.co.uk/bmrs/api/v1",
    "octopus": "https://api.octopus.energy/v1",
}

DAYS = {
    "normal": date(2025, 6, 11),
    "spring_dst": date(2025, 3, 30),
    "autumn_dst": date(2025, 10, 26),
}

AGILE_PRODUCT = "AGILE-24-10-01"
AGILE_REGION = "C"


@dataclass(frozen=True)
class Request:
    label: str
    source: str
    path: str
    params: dict[str, str] = field(default_factory=dict)


def london_day_utc(d: date) -> tuple[datetime, datetime]:
    """UTC [start, end) of a London settlement day; 23h/25h on DST days."""
    start = datetime.combine(d, time(0), LONDON)
    end = datetime.combine(d + timedelta(days=1), time(0), LONDON)
    return start.astimezone(UTC), end.astimezone(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def requests() -> list[Request]:
    reqs: list[Request] = []
    for kind, d in DAYS.items():
        start, end = london_day_utc(d)
        reqs += [
            Request(f"system_prices_{kind}", "elexon", f"/balancing/settlement/system-prices/{d}"),
            Request(
                f"mid_{kind}",
                "elexon",
                "/datasets/MID",
                {"from": iso(start), "to": iso(end - timedelta(minutes=30))},
            ),
            Request(
                f"fuelhh_{kind}",
                "elexon",
                "/datasets/FUELHH",
                {"settlementDateFrom": str(d), "settlementDateTo": str(d)},
            ),
            Request(
                f"agile_{kind}",
                "octopus",
                f"/products/{AGILE_PRODUCT}/electricity-tariffs/"
                f"E-1R-{AGILE_PRODUCT}-{AGILE_REGION}/standard-unit-rates/",
                {"period_from": iso(start), "period_to": iso(end)},
            ),
        ]
        # Forecast vintages: publish windows are capped at 1 day per request. Record the two
        # London days before the target (D-2, D-1) so every vintage visible at D-1 09:00 is present.
        for back in (2, 1):
            p_start, p_end = london_day_utc(d - timedelta(days=back))
            window = {"publishDateTimeFrom": iso(p_start), "publishDateTimeTo": iso(p_end)}
            for ds in ("NDF", "WINDFOR"):
                reqs.append(
                    Request(
                        f"{ds.lower()}_{kind}_pub_d-{back}", "elexon", f"/datasets/{ds}", window
                    )
                )

    n_start, n_end = london_day_utc(DAYS["normal"])
    reqs += [
        Request(
            "temp_normal",
            "elexon",
            "/datasets/TEMP",
            {"publishDateTimeFrom": iso(n_start), "publishDateTimeTo": iso(n_end)},
        ),
        Request("octopus_products", "octopus", "/products/", {"available_at": iso(n_start)}),
        Request("octopus_agile_product", "octopus", f"/products/{AGILE_PRODUCT}/"),
    ]
    return reqs
