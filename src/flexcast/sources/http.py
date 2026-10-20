"""Shared HTTP plumbing: GET-with-retry, time-window chunking, row validation."""

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class UpstreamError(RuntimeError):
    pass


def get_json(
    client: httpx.Client,
    url: str,
    params: dict[str, str] | None = None,
    *,
    attempts: int = 5,
    backoff_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET and decode JSON, retrying transient failures with exponential backoff.

    Retries on network errors and 429/5xx (honouring Retry-After); other 4xx fail immediately
    because retrying a bad request can't succeed.
    """
    for attempt in range(1, attempts + 1):
        retry_after: float | None = None
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError as e:
            reason = f"{type(e).__name__}: {e}"
        else:
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()
                return resp.json()
            reason = f"HTTP {resp.status_code}"
            header = resp.headers.get("Retry-After", "")
            retry_after = float(header) if header.isdigit() else None
        if attempt == attempts:
            raise UpstreamError(f"GET {url} {params} failed after {attempts} attempts: {reason}")
        delay = retry_after if retry_after is not None else backoff_s * 2 ** (attempt - 1)
        log.warning(
            "GET %s failed (%s); retry %d/%d in %.1fs", url, reason, attempt, attempts, delay
        )
        sleep(delay)
    raise AssertionError("unreachable")


def windows(start: datetime, end: datetime, max_len: timedelta) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into consecutive windows no longer than max_len."""
    out = []
    while start < end:
        stop = min(start + max_len, end)
        out.append((start, stop))
        start = stop
    return out


def iso_minute(dt: datetime) -> str:
    """Upstream query format, e.g. 2025-06-10T23:00Z."""
    if dt.tzinfo is None:
        raise ValueError(f"timestamp must be tz-aware: {dt!r}")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


@dataclass
class Parsed[T: BaseModel]:
    """Validated rows plus rejects (raw row + error) so failures are kept, not silently dropped."""

    rows: list[T] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: "Parsed[T]") -> None:
        self.rows += other.rows
        self.rejects += other.rejects


def parse_rows[T: BaseModel](model: type[T], raw: Iterable[dict[str, Any]]) -> Parsed[T]:
    out: Parsed[T] = Parsed()
    for r in raw:
        try:
            out.rows.append(model.model_validate(r))
        except ValidationError as e:
            out.rejects.append({"error": str(e), "raw": r})
    return out
