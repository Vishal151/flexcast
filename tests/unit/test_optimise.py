import numpy as np
import pandas as pd
import pytest

from flexcast.optimise import DT_H, Battery, optimise, settle
from flexcast.settlement import period_starts

TOL = 1e-6


def day_prices(values: list[float] | np.ndarray, day: str = "2025-06-11") -> pd.Series:
    idx = pd.DatetimeIndex(period_starts(pd.Timestamp(day).date()))
    assert len(values) == len(idx), "length must match the day's period count"
    return pd.Series(values, index=idx, name="price_gbp_per_mwh", dtype=float)


def spiky_prices(n: int = 48, seed: int = 0) -> np.ndarray:
    """Evening peak, midday dip, noise and a few negative/spike periods."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 70 + 40 * np.sin((t - 22) / n * 2 * np.pi) + rng.normal(0, 15, n)
    base[rng.choice(n, 3, replace=False)] = rng.uniform(-60, -5, 3)
    base[rng.choice(n, 2, replace=False)] = rng.uniform(250, 900, 2)
    return base


# --- shape and contract ----------------------------------------------------------------------


@pytest.mark.parametrize(("day", "n"), [("2025-06-11", 48), ("2025-03-30", 46), ("2025-10-26", 50)])
def test_schedule_covers_every_period(day: str, n: int) -> None:
    prices = day_prices(spiky_prices(n), day)
    s = optimise(prices)
    assert s.frame.index.equals(prices.index)
    assert {"price_gbp_per_mwh", "charge_kw", "discharge_kw", "soc_kwh"} <= set(s.frame.columns)


def test_objective_equals_settled_value_on_own_prices() -> None:
    prices, b = day_prices(spiky_prices()), Battery()
    s = optimise(prices, b)
    assert s.objective_gbp == pytest.approx(settle(s.frame, prices, b), abs=TOL)


def test_rejects_naive_index() -> None:
    prices = day_prices(spiky_prices())
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    with pytest.raises(ValueError):
        optimise(prices)


# --- physical constraints --------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_power_and_soc_bounds(seed: int) -> None:
    b = Battery()
    f = optimise(day_prices(spiky_prices(seed=seed)), b).frame
    assert f["charge_kw"].between(-TOL, b.power_kw + TOL).all()
    assert f["discharge_kw"].between(-TOL, b.power_kw + TOL).all()
    assert f["soc_kwh"].between(b.soc_min_kwh - TOL, b.capacity_kwh + TOL).all()


@pytest.mark.parametrize("seed", range(5))
def test_energy_balance_every_period(seed: int) -> None:
    b = Battery()
    f = optimise(day_prices(spiky_prices(seed=seed)), b).frame
    prev = np.concatenate([[b.initial_soc_kwh], f["soc_kwh"].to_numpy()[:-1]])
    expected = prev + b.eta * f["charge_kw"] * DT_H - f["discharge_kw"] * DT_H / b.eta
    np.testing.assert_allclose(f["soc_kwh"], expected, atol=TOL)


@pytest.mark.parametrize("terminal", [None, 0.0, 10.0])
def test_terminal_soc(terminal: float | None) -> None:
    b = Battery(terminal_soc_kwh=terminal)
    f = optimise(day_prices(spiky_prices()), b).frame
    assert f["soc_kwh"].iloc[-1] == pytest.approx(b.terminal, abs=TOL)


# --- economics -------------------------------------------------------------------------------


def test_flat_prices_mean_no_cycling() -> None:
    """With losses and degradation, cycling at a flat price can only lose money."""
    f = optimise(day_prices([80.0] * 48)).frame
    assert f["charge_kw"].abs().max() < TOL
    assert f["discharge_kw"].abs().max() < TOL


def two_price(low: float, high: float) -> pd.Series:
    """4 cheap periods, then 4 expensive, then 40 at the cheap price (a 48-period day)."""
    return day_prices([low] * 4 + [high] * 4 + [low] * 40)


def test_hand_solvable_ideal_battery() -> None:
    """No losses, no degradation, starts/ends empty. 4 periods at 5 kW * 0.5 h = 10 kWh fills
    it exactly at £0; sell 10 kWh at £100/MWh = £0.10/kWh -> £1.00."""
    b = Battery(round_trip_efficiency=1.0, degradation_gbp_per_kwh=0.0, initial_soc_kwh=0.0)
    s = optimise(two_price(0.0, 100.0), b)
    assert s.objective_gbp == pytest.approx(1.00, abs=1e-4)


def test_hand_solvable_lossy_battery() -> None:
    """eta = 0.9 each way, 1p/kWh degradation, starts/ends empty, buy at £20, sell at £100/MWh.
    Charge: power-limited, 4 * 2.5 kWh = 10 kWh from grid -> SoC 0.9 * 10 = 9 kWh.
    Discharge: 9 kWh * 0.9 = 8.1 kWh to grid.
    £ = 8.1 * 0.10 - 10 * 0.02 - 0.01 * (10 + 8.1) = 0.81 - 0.20 - 0.181 = 0.429."""
    b = Battery(round_trip_efficiency=0.81, degradation_gbp_per_kwh=0.01, initial_soc_kwh=0.0)
    s = optimise(two_price(20.0, 100.0), b)
    assert s.objective_gbp == pytest.approx(0.429, abs=1e-4)


def test_lower_efficiency_never_earns_more() -> None:
    prices = day_prices(spiky_prices())
    v90 = optimise(prices, Battery(round_trip_efficiency=0.9)).objective_gbp
    v80 = optimise(prices, Battery(round_trip_efficiency=0.8)).objective_gbp
    assert v80 <= v90 + TOL


# --- the LP vs MILP decision -----------------------------------------------------------------


def test_no_simultaneous_charge_and_discharge_at_very_negative_prices() -> None:
    """A full battery at -£1000/MWh. A pure LP can 'charge and discharge at once': SoC stays
    put, but the round-trip losses let it import net energy and get paid for it, which a real
    battery can't do. If you choose a pure LP instead, replace this test with one that documents
    and bounds that behaviour, and note why in the journal."""
    b = Battery(initial_soc_kwh=10.0)
    f = optimise(day_prices([-1000.0] * 48), b).frame
    both = (f["charge_kw"] > TOL) & (f["discharge_kw"] > TOL)
    assert not both.any()


# --- separate buy/sell prices (risk-aware scheduling) ----------------------------------------


def test_sell_prices_default_to_prices() -> None:
    prices = day_prices(spiky_prices())
    a, b = optimise(prices), optimise(prices, sell_prices=prices)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.objective_gbp == pytest.approx(b.objective_gbp)


def test_pessimistic_spread_wider_than_signal_means_no_trading() -> None:
    """A £1200 buy/sell gap exceeds the largest spread in spiky_prices (spikes reach £900)."""
    prices = day_prices(spiky_prices())
    f = optimise(prices + 600, sell_prices=prices - 600).frame
    assert f["charge_kw"].abs().max() < TOL and f["discharge_kw"].abs().max() < TOL


def test_hand_solvable_buy_sell_spread() -> None:
    """Ideal battery as test_hand_solvable_ideal_battery, but buying costs £10/MWh more and
    selling earns £10/MWh less: 10 kWh * (0.09 - 0.01) = £0.80."""
    b = Battery(round_trip_efficiency=1.0, degradation_gbp_per_kwh=0.0, initial_soc_kwh=0.0)
    p = two_price(0.0, 100.0)
    s = optimise(p + 10, b, sell_prices=p - 10)
    assert s.objective_gbp == pytest.approx(0.80, abs=1e-4)


def test_sell_prices_must_align() -> None:
    prices = day_prices(spiky_prices())
    with pytest.raises(ValueError, match="sell_prices"):
        optimise(prices, sell_prices=prices.iloc[:-1])
