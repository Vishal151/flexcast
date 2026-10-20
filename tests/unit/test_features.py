from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flexcast.features import FEATURE_COLUMNS, build, decision_time_for
from flexcast.history import History
from flexcast.settlement import LONDON, period_starts
from flexcast.storage import Storage
from tests.unit.synthetic import encode_price, encode_publish

HistoryFactory = Callable[[date, date], History]

NORMAL, AUTUMN, AFTER_SPRING = date(2025, 6, 11), date(2025, 10, 26), date(2025, 4, 1)
TARGETS = [NORMAL, AUTUMN, AFTER_SPRING, date(2025, 3, 30)]


def history_for(factory: HistoryFactory, target: date) -> History:
    return factory(target - timedelta(days=12), target + timedelta(days=1))


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


# --- leakage ---------------------------------------------------------------------------------


def poison_future(h: History, decision_time: datetime) -> History:
    """Replace every value knowable at/after decision_time with garbage."""

    def bad(df: pd.DataFrame, col: str) -> pd.DataFrame:
        df = df.copy()
        df.loc[df["available_at"] >= decision_time, col] = 1e9
        return df

    return History(
        prices=bad(h.prices, "price"),
        mid=bad(h.mid, "price"),
        ndf=bad(h.ndf, "demand"),
        windfor=bad(h.windfor, "generation"),
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("at", [time(9), time(6, 15), time(23, 59)])
def test_leakage_poisoned_future_does_not_change_features(
    synthetic_history: HistoryFactory, target: date, at: time
) -> None:
    """The required leakage test: corrupt everything published at/after the decision time;
    if any feature changes, it was reading the future."""
    h = history_for(synthetic_history, target)
    dt = decision_time_for(target, at)
    clean = build(h, dt, target).X
    poisoned = build(poison_future(h, dt), dt, target).X
    pd.testing.assert_frame_equal(clean, poisoned)


@pytest.mark.parametrize("target", TARGETS)
def test_leakage_every_source_used_was_available_before_decision(
    synthetic_history: HistoryFactory, target: date
) -> None:
    h = history_for(synthetic_history, target)
    dt = decision_time_for(target)
    f = build(h, dt, target)
    assert set(f.provenance) == {"prices", "mid", "ndf", "windfor"}
    for source, t in f.provenance.items():
        assert t < dt, f"{source} used data available at {t}, decision {dt}"


def test_rejects_decision_on_or_after_target_day(synthetic_history: HistoryFactory) -> None:
    h = history_for(synthetic_history, NORMAL)
    with pytest.raises(ValueError, match="not before"):
        build(h, datetime.combine(NORMAL, time(0), LONDON), NORMAL)
    with pytest.raises(ValueError, match="tz-aware"):
        build(h, datetime(2025, 6, 10, 9), NORMAL)  # noqa: DTZ001


# --- decision time ---------------------------------------------------------------------------


def test_decision_time_is_london_wall_clock() -> None:
    assert decision_time_for(NORMAL) == utc("2025-06-10T08:00:00+00:00")  # BST
    assert decision_time_for(date(2025, 1, 15)) == utc("2025-01-14T09:00:00+00:00")  # GMT


# --- correctness -----------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_shape_and_columns(synthetic_history: HistoryFactory, target: date) -> None:
    f = build(history_for(synthetic_history, target), decision_time_for(target), target)
    assert list(f.X.columns) == FEATURE_COLUMNS
    assert list(f.X.index) == period_starts(target)
    assert (f.X["n_periods"] == len(period_starts(target))).all()


def test_price_lags_match_same_london_clock_time(synthetic_history: HistoryFactory) -> None:
    f = build(history_for(synthetic_history, NORMAL), decision_time_for(NORMAL), NORMAL).X
    for ts, (_, row) in zip(pd.DatetimeIndex(f.index), f.iterrows(), strict=True):
        assert row["price_d2"] == encode_price(ts - pd.Timedelta(days=2))
        assert row["price_d7"] == encode_price(ts - pd.Timedelta(days=7))
        assert row["mid_d2"] == encode_price(ts - pd.Timedelta(days=2)) + 0.5  # APX, not N2EX


def test_autumn_repeated_hour_gets_the_same_lag(synthetic_history: HistoryFactory) -> None:
    f = build(history_for(synthetic_history, AUTUMN), decision_time_for(AUTUMN), AUTUMN).X
    # 01:00 London happens at 00:00Z (BST) and 01:00Z (GMT).
    bst, gmt = pd.Timestamp("2025-10-26T00:00Z"), pd.Timestamp("2025-10-26T01:00Z")
    assert f.at[bst, "price_d2"] == f.at[gmt, "price_d2"]


def test_clock_time_missing_on_spring_day_is_nan(synthetic_history: HistoryFactory) -> None:
    # D-2 of 2025-04-01 is 2025-03-30, which has no 01:00-01:59 London.
    f = build(
        history_for(synthetic_history, AFTER_SPRING), decision_time_for(AFTER_SPRING), AFTER_SPRING
    ).X
    local = pd.DatetimeIndex(f.index).tz_convert(LONDON)
    missing = f["price_d2"][local.hour == 1]
    assert len(missing) == 2 and missing.isna().all()
    assert f["price_d2"][local.hour != 1].notna().all()


def test_latest_price_is_last_available_before_decision(synthetic_history: HistoryFactory) -> None:
    dt = decision_time_for(NORMAL)  # 08:00Z; prices available 1 h after start -> last is 06:30Z
    f = build(history_for(synthetic_history, NORMAL), dt, NORMAL).X
    assert (f["price_latest"] == encode_price(pd.Timestamp("2025-06-10T06:30Z"))).all()
    assert (f["price_latest_age_h"] == 1.5).all()


def test_ndf_uses_latest_vintage_per_period(synthetic_history: HistoryFactory) -> None:
    dt = decision_time_for(NORMAL)  # 08:00Z; last NDF publish strictly before is 07:30Z
    f = build(history_for(synthetic_history, NORMAL), dt, NORMAL).X
    # Synthetic vintages reach 48 h ahead, so 07:30Z covers the whole target day.
    assert (f["ndf"] == encode_publish(pd.Timestamp("2025-06-10T07:30Z"))).all()
    assert (f["ndf_age_h"] == 0.5).all()


def test_ndf_stitches_vintages_with_partial_coverage(synthetic_history: HistoryFactory) -> None:
    """Real NDF: the latest run covers only part of D, an earlier run covers the rest."""
    h = history_for(synthetic_history, NORMAL)
    starts = period_starts(NORMAL)
    early, late = pd.Timestamp("2025-06-10T05:00Z"), pd.Timestamp("2025-06-10T07:45Z")
    ndf = pd.DataFrame(
        [(early, s, 1) for s in starts] + [(late, s, 2) for s in starts[10:]],
        columns=["publish_time", "start_time", "demand"],
    )
    h2 = History.from_frames(
        h.prices.rename(columns={"price": "system_sell_price"}).assign(
            created_date_time=h.prices["start_time"]
        ),
        h.mid.assign(data_provider="APXMIDP"),
        ndf,
        h.windfor,
    )
    f = build(h2, decision_time_for(NORMAL), NORMAL).X
    assert f["ndf"].tolist() == [1.0] * 10 + [2.0] * 38
    np.testing.assert_allclose(f["ndf_age_h"], [3.0] * 10 + [0.25] * 38)


def test_wind_hourly_value_fills_both_half_hours(synthetic_history: HistoryFactory) -> None:
    f = build(history_for(synthetic_history, NORMAL), decision_time_for(NORMAL), NORMAL).X
    w = f["wind"].to_numpy()
    assert (w[0::2] == w[1::2]).all()
    assert (f["residual_demand"] == f["ndf"] - f["wind"]).all()


def test_calendar(synthetic_history: HistoryFactory) -> None:
    xmas = date(2025, 12, 25)
    f = build(history_for(synthetic_history, xmas), decision_time_for(xmas), xmas).X
    assert (f["is_holiday"] == 1).all() and (f["dow"] == 3).all() and (f["is_weekend"] == 0).all()
    assert f["hour"].iloc[0] == 0.0 and f["hour"].iloc[-1] == 23.5


def test_actual_prices_not_affected_by_as_of(synthetic_history: HistoryFactory) -> None:
    h = history_for(synthetic_history, NORMAL)
    actual = h.actual_prices(NORMAL)
    assert len(actual) == 48
    assert actual.iloc[0] == encode_price(pd.Timestamp(period_starts(NORMAL)[0]))


def test_history_from_frames_keeps_latest_price_revision() -> None:
    t = pd.Timestamp("2025-06-10T23:00Z")
    prices = pd.DataFrame(
        {
            "start_time": [t, t],
            "created_date_time": [t + pd.Timedelta("1h"), t + pd.Timedelta("25h")],
            "system_sell_price": [10.0, 12.0],
        }
    )
    no_time = pd.Series([], dtype="datetime64[us, UTC]")
    empty_mid = pd.DataFrame({"start_time": no_time, "data_provider": [], "price": []})
    empty_fc = pd.DataFrame(
        {"publish_time": no_time, "start_time": no_time, "demand": [], "generation": []}
    )
    h = History.from_frames(prices, empty_mid, empty_fc, empty_fc)
    assert h.prices["price"].tolist() == [12.0]


def test_availability_rules_are_pinned(synthetic_history: HistoryFactory) -> None:
    """The poisoning test trusts available_at; this pins the rules themselves (SPEC §2)."""
    h = history_for(synthetic_history, NORMAL)
    hour = pd.Timedelta(hours=1)
    assert (h.prices["available_at"] == h.prices["start_time"] + hour).all()
    assert (h.mid["available_at"] == h.mid["start_time"] + hour).all()
    assert (h.ndf["available_at"] == h.ndf["publish_time"]).all()
    assert (h.windfor["available_at"] == h.windfor["publish_time"]).all()


def test_history_load_from_empty_storage(tmp_path: Path) -> None:
    """A fresh deployment has no data yet; loading must give an empty history, not crash."""
    h = History.load(Storage(f"file://{tmp_path}"))
    assert h.prices.empty and h.mid.empty and h.ndf.empty and h.windfor.empty
    assert h.actual_prices(NORMAL).empty
    assert h.as_of("ndf", decision_time_for(NORMAL), NORMAL).empty
