"""Pull -> validate -> write partitions. Idempotent: re-running a day merges on a natural key and
rewrites a byte-identical file.

Layout: raw/<dataset>/date=<London date>/part.parquet;
rejects go to rejects/<dataset>/date=<d>/rejects.jsonl.
The partition date is the settlement date for outturns and the London date of publish_time /
valid_from for forecasts and tariffs. Each ingest_day(d) writes only partition d, so days can run
in parallel without two writers touching the same file.
"""

import json
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel

from flexcast.settlement import LONDON, london_day_bounds
from flexcast.sources.elexon import ElexonClient
from flexcast.sources.http import Parsed
from flexcast.sources.neso import NesoClient
from flexcast.sources.octopus import OctopusClient
from flexcast.storage import Storage

log = logging.getLogger(__name__)

Records = tuple[list[dict[str, Any]], list[dict[str, Any]]]  # (rows, rejects)


@dataclass(frozen=True)
class Dataset:
    name: str
    key: tuple[str, ...]  # natural key incl. publish/created time, so revisions are kept
    partition_col: str


DATASETS = {
    d.name: d
    for d in [
        Dataset("system_prices", ("start_time", "created_date_time"), "settlement_date"),
        Dataset("mid", ("start_time", "data_provider"), "settlement_date"),
        Dataset("fuelhh", ("start_time", "fuel_type", "publish_time"), "settlement_date"),
        Dataset("ndf", ("publish_time", "start_time"), "publish_time"),
        Dataset("windfor", ("publish_time", "start_time"), "publish_time"),
        Dataset("agile", ("valid_from", "product_code", "region"), "valid_from"),
        Dataset("embedded", ("publish_time", "start_time"), "publish_time"),
    ]
}


def default_range(now: datetime) -> tuple[date, date]:
    """Yesterday and today, as London days."""
    today = now.astimezone(LONDON).date()
    return today - timedelta(days=1), today


def merge(existing: pd.DataFrame | None, new: pd.DataFrame, key: tuple[str, ...]) -> pd.DataFrame:
    """Union, drop duplicate keys (newest fetch wins), sort by key: same input -> same output."""
    df = new if existing is None else pd.concat([existing, new], ignore_index=True)
    df = df.drop_duplicates(subset=list(key), keep="last")
    return df.sort_values(list(key), kind="mergesort").reset_index(drop=True)


def _records[T: BaseModel](parsed: Parsed[T], **extra: str) -> Records:
    return [{**r.model_dump(), **extra} for r in parsed.rows], parsed.rejects


def _to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in df.columns:
        if isinstance(df[col].iloc[0], datetime):
            df[col] = pd.to_datetime(df[col], utc=True)  # normalise tzinfo to plain UTC
    return df


def _partition_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.astimezone(LONDON).date()
    if isinstance(value, date):
        return value
    raise TypeError(f"cannot derive partition date from {value!r}")


class Ingester:
    def __init__(
        self,
        storage: Storage,
        elexon: ElexonClient,
        octopus: OctopusClient,
        region: str,
        neso: NesoClient,
    ) -> None:
        self.storage = storage
        self.elexon = elexon
        self.octopus = octopus
        self.neso = neso
        self.region = region
        self._tariffs: dict[str, str] = {}
        self._fetchers: dict[str, Callable[[date], Records]] = {
            "system_prices": lambda d: _records(self.elexon.system_prices(d)),
            "mid": lambda d: _records(self.elexon.mid(*london_day_bounds(d))),
            "fuelhh": lambda d: _records(self.elexon.fuelhh(d, d)),
            "ndf": lambda d: _records(self.elexon.ndf(*london_day_bounds(d))),
            "windfor": lambda d: _records(self.elexon.windfor(*london_day_bounds(d))),
            "agile": self._agile,
            "embedded": self._embedded,
        }

    def _embedded(self, d: date) -> Records:
        parsed = self.neso.embedded_forecasts(*london_day_bounds(d))
        # period_end_gmt only exists to cross-check start_time; don't store it.
        return [r.model_dump(exclude={"period_end_gmt"}) for r in parsed.rows], parsed.rejects

    def _agile(self, d: date) -> Records:
        start, end = london_day_bounds(d)
        product = self.octopus.find_agile_product(start)
        if product.code not in self._tariffs:
            self._tariffs[product.code] = self.octopus.tariff_code(product.code, self.region)
        parsed = self.octopus.agile_rates(product.code, self._tariffs[product.code], start, end)
        return _records(parsed, product_code=product.code, region=self.region)

    def ingest_day(
        self, d: date, datasets: Iterable[str] | None = None
    ) -> dict[str, tuple[int, int]]:
        """Fetch and write partition `d` of each dataset. Returns {dataset: (rows, rejects)}.

        A failing dataset doesn't stop the others; failures are raised together at the end.
        """
        result: dict[str, tuple[int, int]] = {}
        errors: list[Exception] = []
        for name in datasets or DATASETS:
            try:
                rows, rejects = self._fetchers[name](d)
                result[name] = self._write(DATASETS[name], d, rows, rejects)
            except Exception as e:
                log.error("ingest %s %s failed: %s", name, d, e)
                e.add_note(f"dataset={name} date={d}")
                errors.append(e)
        if errors:
            raise ExceptionGroup(f"ingest {d}: {len(errors)} dataset(s) failed", errors)
        return result

    def _write(
        self,
        ds: Dataset,
        d: date,
        rows: list[dict[str, Any]],
        rejects: list[dict[str, Any]],
    ) -> tuple[int, int]:
        # Upstream windows are inclusive at both ends; keep only rows that belong to partition d.
        rows = [r for r in rows if _partition_date(r[ds.partition_col]) == d]
        part = f"date={d.isoformat()}"
        if rejects:
            lines = [json.dumps(r, sort_keys=True, default=str) for r in rejects]
            self.storage.write_text(
                "\n".join(lines) + "\n", f"rejects/{ds.name}/{part}/rejects.jsonl"
            )
            log.warning("%s %s: %d rejected rows", ds.name, d, len(rejects))
        if rows:
            path = f"raw/{ds.name}/{part}/part.parquet"
            existing = self.storage.read_parquet(path) if self.storage.exists(path) else None
            self.storage.write_parquet(merge(existing, _to_frame(rows), ds.key), path)
        return len(rows), len(rejects)

    def run(
        self, start: date, end: date, datasets: Iterable[str] | None = None, concurrency: int = 4
    ) -> list[date]:
        """Ingest London days start..end inclusive. Returns the days that had failures."""
        names = list(datasets or DATASETS)
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        failed: list[date] = []

        def one(d: date) -> None:
            try:
                counts = self.ingest_day(d, names)
                log.info("ingested %s %s", d, counts)
            except ExceptionGroup:
                failed.append(d)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(one, days))
        log.info(
            "ingest %s..%s at %s: %d days, %d failed",
            start,
            end,
            datetime.now(UTC),
            len(days),
            len(failed),
        )
        return sorted(failed)
