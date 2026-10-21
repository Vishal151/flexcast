from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flexcast.features import OUTPUT_COLUMNS, build, decision_time_for
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

    embedded = bad(bad(bad(h.embedded, "solar_mw"), "solar_capacity_mw"), "wind_mw")
    return History(
        prices=bad(h.prices, "price"),
        mid=bad(h.mid, "price"),
        ndf=bad(h.ndf, "demand"),
        windfor=bad(h.windfor, "generation"),
        embedded=embedded,
        day_ahead=bad(h.day_ahead, "price"),
        availability=bad(h.availability, "output_usable"),
        gas=bad(h.gas, "price"),
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
    assert set(f.provenance) == {
        "prices",
        "mid",
        "ndf",
        "windfor",
        "embedded",
        "day_ahead",
        "availability",
        "gas",
    }
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
    assert list(f.X.columns) == OUTPUT_COLUMNS
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
    two_hours = pd.Timedelta(hours=2)
    assert (h.embedded["available_at"] == h.embedded["publish_time"] + two_hours).all()


def test_history_load_from_empty_storage(tmp_path: Path) -> None:
    """A fresh deployment has no data yet; loading must give an empty history, not crash."""
    h = History.load(Storage(f"file://{tmp_path}"))
    assert h.prices.empty and h.mid.empty and h.ndf.empty and h.windfor.empty
    assert h.actual_prices(NORMAL).empty
    assert h.as_of("ndf", decision_time_for(NORMAL), NORMAL).empty


def test_solar_uses_latest_vintage_at_least_two_hours_old(
    synthetic_history: HistoryFactory,
) -> None:
    dt = decision_time_for(NORMAL)  # 08:00Z: vintages count from publish + 2 h -> last is 05:12Z
    f = build(history_for(synthetic_history, NORMAL), dt, NORMAL).X
    assert (f["solar"] == encode_publish(pd.Timestamp("2025-06-10T05:12Z"))).all()
    np.testing.assert_allclose(f["embedded_age_h"], 2.8)
    np.testing.assert_allclose(f["solar_capacity_factor"], f["solar"] / 20_000)
    assert (f["embedded_wind"] == 1_000).all()


def test_history_without_embedded_data_gives_nan_solar(synthetic_history: HistoryFactory) -> None:
    h = history_for(synthetic_history, NORMAL)
    h2 = replace(h, embedded=h.embedded.iloc[:0])
    f = build(h2, decision_time_for(NORMAL), NORMAL)
    assert f.X["solar"].isna().all() and "embedded" not in f.provenance


def test_day_ahead_lags_use_yesterday_not_the_target_day(synthetic_history: HistoryFactory) -> None:
    """At D-1 09:00, D-1's auction prices are known (published D-2 10:00); D's aren't yet."""
    f = build(history_for(synthetic_history, NORMAL), decision_time_for(NORMAL), NORMAL).X
    for ts, (_, row) in zip(pd.DatetimeIndex(f.index), f.iterrows(), strict=True):
        hour_start = ts.floor("h")
        assert row["da_d1"] == encode_price(hour_start - pd.Timedelta(days=1)) + 500
        assert row["da_d7"] == encode_price(hour_start - pd.Timedelta(days=7)) + 500
    assert (f["da_d1_max"] >= f["da_d1"]).all() and (f["da_d1_min"] <= f["da_d1"]).all()


def test_day_ahead_availability_rule_is_13_00_on_the_day_before(
    synthetic_history: HistoryFactory,
) -> None:
    h = history_for(synthetic_history, NORMAL)
    row = h.day_ahead[h.day_ahead["start_time"] == pd.Timestamp("2025-06-11T12:30Z")].iloc[0]
    assert row["available_at"] == pd.Timestamp("2025-06-10T12:00Z")  # 13:00 BST on D-1
    assert row["price"] == encode_price(pd.Timestamp("2025-06-11T12:00Z")) + 500  # hour's price


def test_actual_day_ahead_prices_are_half_hourly(synthetic_history: HistoryFactory) -> None:
    h = history_for(synthetic_history, AUTUMN)
    actual = h.actual_prices(AUTUMN, "day_ahead")
    assert list(actual.index) == period_starts(AUTUMN)  # 50 half-hours
    assert (actual.to_numpy()[0::2] == actual.to_numpy()[1::2]).all()


# --- candidate inputs: gas and plant availability ---------------------------------------------


def test_gas_availability_rule_is_13_00_the_day_after_the_gas_day(
    synthetic_history: HistoryFactory,
) -> None:
    h = history_for(synthetic_history, NORMAL)
    row = h.gas[h.gas["gas_day"] == date(2025, 6, 8)].iloc[0]
    assert row["available_at"] == pd.Timestamp("2025-06-09T12:00Z")  # 13:00 BST on day + 1
    assert row["price"] == pytest.approx(date(2025, 6, 8).toordinal() % 97)  # p/kWh -> £/MWh


def test_gas_at_d1_09_00_is_the_gas_day_three_days_before(
    synthetic_history: HistoryFactory,
) -> None:
    """D-2's price appears at 13:00 on D-1, after the 09:00 decision, so the latest is D-3."""
    f = build(history_for(synthetic_history, NORMAL), decision_time_for(NORMAL), NORMAL).X
    expected = [(NORMAL - timedelta(days=k)).toordinal() % 97 for k in range(3, 10)]
    np.testing.assert_allclose(f["gas_price"], expected[0])
    np.testing.assert_allclose(f["gas_price_7d"], np.mean(expected))


def test_availability_uses_latest_vintage_covering_the_target_day(
    synthetic_history: HistoryFactory,
) -> None:
    """Vintages cover publish date + 2 days onward, so for D the last one is published at 23:00
    BST on D-2 (22:00Z): the 09:00 D-1 vintage only starts at D+1."""
    h = history_for(synthetic_history, NORMAL)
    f = build(h, decision_time_for(NORMAL), NORMAL).X
    assert (f["avail_ccgt"] == encode_publish(pd.Timestamp("2025-06-09T22:00Z"))).all()
    assert (f["avail_nuclear"] == 4_000).all() and (f["avail_interconnector"] == 2_000).all()
    assert (f["avail_dispatchable"] == f["avail_ccgt"] + 4_000).all()  # excludes wind
    expected = f["avail_dispatchable"] + 2_000 + f["wind"] - f["ndf"]
    pd.testing.assert_series_equal(f["margin"], expected, check_names=False)
    assert (h.availability["available_at"] == h.availability["publish_time"]).all()


def test_history_without_candidate_data_gives_nan(synthetic_history: HistoryFactory) -> None:
    h = history_for(synthetic_history, NORMAL)
    h2 = replace(h, availability=h.availability.iloc[:0], gas=h.gas.iloc[:0])
    f = build(h2, decision_time_for(NORMAL), NORMAL)
    cols = ["gas_price", "gas_price_7d", "avail_ccgt", "avail_dispatchable", "margin"]
    assert f.X[cols].isna().all().all()
    assert "gas" not in f.provenance and "availability" not in f.provenance


def test_availability_ignores_vintages_published_after_an_earlier_decision(
    synthetic_history: HistoryFactory,
) -> None:
    """At D-1 09:00 every vintage covering D is already past (they stop at D-2), so this uses an
    earlier decision to show the availability filter is applied, not just luckily unneeded."""
    h = history_for(synthetic_history, NORMAL)
    dt = datetime(2025, 6, 9, 12, tzinfo=LONDON)  # D-2 12:00 BST = 11:00Z
    f = build(h, dt, NORMAL).X
    assert (f["avail_ccgt"] == encode_publish(pd.Timestamp("2025-06-09T10:00Z"))).all()
