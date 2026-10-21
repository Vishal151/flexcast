"""The upstream requests Stage 0 explores and records. Shared by explore_api.py and
record_fixtures.py so the two cannot drift. Throwaway: Stage 1 replaces this with real clients."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")

BASE_URLS = {
    "elexon": "https://data.elexon.co.uk/bmrs/api/v1",
    "octopus": "https://api.octopus.energy/v1",
    "neso": "https://api.neso.energy/api/3/action",
    "gas": "https://data.nationalgas.com/api",
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
                f"octopus_products_{kind}", "octopus", "/products/", {"available_at": iso(start)}
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
        Request("octopus_agile_product", "octopus", f"/products/{AGILE_PRODUCT}/"),
    ]
    return reqs + neso_requests() + candidate_requests()


def candidate_requests() -> list[Request]:
    """Gas SAP and plant availability (docs/candidate-sources.md): the windows a feature build
    for the normal day needs (availability vintages published on D-2; gas days D-9..D-3)."""
    from flexcast.sources.gas import sap_params

    d = DAYS["normal"]
    p_start, p_end = london_day_utc(d - timedelta(days=2))
    return [
        Request(
            "availability_normal_pub_d-2",
            "elexon",
            "/datasets/FOU2T14D",
            {"publishDateTimeFrom": iso(p_start), "publishDateTimeTo": iso(p_end)},
        ),
        Request(
            "gas_sap_normal",
            "gas",
            "/find-gas-data-download",
            sap_params(d - timedelta(days=9), d - timedelta(days=3)),
        ),
        Request(  # what ingest asks for: one gas day per partition
            "gas_sap_one_day",
            "gas",
            "/find-gas-data-download",
            sap_params(d - timedelta(days=6), d - timedelta(days=6)),
        ),
    ]


def neso_requests() -> list[Request]:
    """Embedded solar/wind vintages for the publish windows features need (Stage 2c)."""
    from flexcast.sources.neso import PACKAGE, NesoClient, embedded_sql

    archives = NesoClient(BASE_URLS["neso"]).archive_resources()
    reqs = [Request("neso_embedded_package", "neso", "/package_show", {"id": PACKAGE})]
    for label, d in [
        ("normal_pub_d-2", date(2025, 6, 9)),
        ("normal_pub_d-1", date(2025, 6, 10)),
        ("autumn_dst_pub_d-1", date(2025, 10, 25)),
    ]:
        start, end = london_day_utc(d)
        sql = embedded_sql(archives[start.year], start, end)
        reqs.append(
            Request(f"neso_embedded_{label}", "neso", "/datastore_search_sql", {"sql": sql})
        )
    from flexcast.sources.neso import DAY_AHEAD_PACKAGE, day_ahead_sql

    da_resource = "4f27eea5-7038-4f73-9740-e3e4ad47c26a"  # recorded resource; client discovers it
    reqs.append(
        Request("neso_day_ahead_package", "neso", "/package_show", {"id": DAY_AHEAD_PACKAGE})
    )
    for kind, d in DAYS.items():
        start, end = london_day_utc(d)
        sql = day_ahead_sql(da_resource, start.date(), (end - timedelta(seconds=1)).date())
        reqs.append(
            Request(f"neso_day_ahead_{kind}", "neso", "/datastore_search_sql", {"sql": sql})
        )
    return reqs
