"""The one place settlement date/period is derived from time.

A settlement day is a London calendar day; period 1 starts at London midnight and each period is
30 minutes, so a day has 46/48/50 periods. We always go UTC -> (date, period), counting elapsed
UTC time since London midnight; never period -> time via `hour * 2 + 1`.
"""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
PERIOD = timedelta(minutes=30)


def london_day_bounds(d: date) -> tuple[datetime, datetime]:
    """UTC [start, end) of a London settlement day."""
    start = datetime.combine(d, time(0), LONDON).astimezone(UTC)
    end = datetime.combine(d + timedelta(days=1), time(0), LONDON).astimezone(UTC)
    return start, end


def to_settlement(ts: datetime) -> tuple[date, int]:
    """Settlement (date, period) of the half-hour starting at `ts`."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"timestamp must be tz-aware: {ts!r}")
    ts = ts.astimezone(UTC)
    if ts.minute % 30 or ts.second or ts.microsecond:
        raise ValueError(f"timestamp is not on a half-hour boundary: {ts.isoformat()}")
    d = ts.astimezone(LONDON).date()
    start, _ = london_day_bounds(d)
    return d, (ts - start) // PERIOD + 1


def period_starts(d: date) -> list[datetime]:
    """UTC start time of every settlement period on London day `d`."""
    start, end = london_day_bounds(d)
    return [start + i * PERIOD for i in range((end - start) // PERIOD)]
