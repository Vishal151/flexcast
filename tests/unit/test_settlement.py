import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from flexcast.settlement import period_starts, to_settlement

FIXTURES = Path(__file__).parents[1] / "fixtures" / "elexon"


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@pytest.mark.parametrize("kind", ["normal", "spring_dst", "autumn_dst"])
def test_matches_elexon_for_every_period(kind: str) -> None:
    """Elexon's own startTime -> (settlementDate, settlementPeriod) mapping is the oracle."""
    rows = json.loads((FIXTURES / f"system_prices_{kind}.json").read_text())["data"]
    for r in rows:
        assert to_settlement(utc(r["startTime"])) == (
            date.fromisoformat(r["settlementDate"]),
            r["settlementPeriod"],
        )


@pytest.mark.parametrize(
    ("d", "n"), [(date(2025, 6, 11), 48), (date(2025, 3, 30), 46), (date(2025, 10, 26), 50)]
)
def test_period_starts_count_and_spacing(d: date, n: int) -> None:
    starts = period_starts(d)
    assert len(starts) == n
    assert all(s.tzinfo is UTC for s in starts)
    assert all(b - a == timedelta(minutes=30) for a, b in zip(starts, starts[1:], strict=False))


@pytest.mark.parametrize("d", [date(2025, 6, 11), date(2025, 3, 30), date(2025, 10, 26)])
def test_round_trip(d: date) -> None:
    for i, s in enumerate(period_starts(d), start=1):
        assert to_settlement(s) == (d, i)


def test_autumn_repeated_local_hour_is_two_distinct_periods() -> None:
    # 01:00-02:00 London happens twice on 2025-10-26: BST (00:00Z) then GMT (01:00Z).
    assert to_settlement(utc("2025-10-26T00:00:00Z")) == (date(2025, 10, 26), 3)
    assert to_settlement(utc("2025-10-26T01:00:00Z")) == (date(2025, 10, 26), 5)


def test_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        to_settlement(datetime(2025, 6, 11, 12, 0))  # noqa: DTZ001


def test_rejects_off_grid_timestamp() -> None:
    with pytest.raises(ValueError, match="half-hour"):
        to_settlement(utc("2025-06-11T12:15:00Z"))


def test_non_utc_aware_input_is_converted() -> None:
    london_noon = datetime.fromisoformat("2025-06-11T12:00:00+01:00")
    assert to_settlement(london_noon) == (date(2025, 6, 11), 25)  # 24 half-hours after midnight
