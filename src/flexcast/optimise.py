"""Battery dispatch optimiser: given half-hourly prices, choose charge/discharge to maximise £.

Conventions
- `prices`: pd.Series in £/MWh, indexed by tz-aware UTC period start times, consecutive 30-min.
  Works for 46/48/50-period days (and any horizon); never assume 48.
- Power in kW at the grid connection; energy per period = power * DT_H.
- Round-trip efficiency is split evenly: eta = sqrt(round_trip_efficiency) on charge and discharge.
  SoC gains eta * charge_kw * DT_H; loses discharge_kw * DT_H / eta.
- Degradation is charged on grid-side throughput: (charge_kw + discharge_kw) * DT_H * £/kWh.
- soc_kwh in the schedule is the state of charge at the END of each period.
"""

import math
from dataclasses import dataclass

import pandas as pd
import pulp

DT_H = 0.5  # hours per settlement period


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float = 10.0
    power_kw: float = 5.0
    round_trip_efficiency: float = 0.90
    soc_min_kwh: float = 0.0
    initial_soc_kwh: float = 5.0
    terminal_soc_kwh: float | None = None  # None: same as initial
    degradation_gbp_per_kwh: float = 0.02
    # Day-ahead auction products are whole hours: hold one power level across both half-hours.
    hourly_blocks: bool = False

    @property
    def eta(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def terminal(self) -> float:
        return self.initial_soc_kwh if self.terminal_soc_kwh is None else self.terminal_soc_kwh


@dataclass(frozen=True)
class Schedule:
    """frame: index = prices.index; columns price_gbp_per_mwh, charge_kw, discharge_kw, soc_kwh."""

    frame: pd.DataFrame
    objective_gbp: float


def settle(frame: pd.DataFrame, prices: pd.Series, battery: Battery) -> float:
    """£ earned by executing a schedule at `prices` (which may differ from the prices it was
    optimised on: that's how a forecast-driven schedule is valued in the backtest)."""
    p_gbp_per_kwh = prices.reindex(frame.index) / 1000
    energy = (frame["discharge_kw"] - frame["charge_kw"]) * DT_H
    throughput = (frame["charge_kw"] + frame["discharge_kw"]) * DT_H
    return float(
        (energy * p_gbp_per_kwh).sum() - battery.degradation_gbp_per_kwh * throughput.sum()
    )


def _validate(prices: pd.Series) -> pd.DatetimeIndex:
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("prices must be indexed by tz-aware timestamps")
    if len(idx) == 0:
        raise ValueError("prices is empty")
    if (idx[1:] - idx[:-1] != pd.Timedelta(minutes=30)).any():
        raise ValueError("prices must be consecutive 30-minute periods")
    if prices.isna().any():
        raise ValueError("prices contains NaN")
    return idx


def _solved(v: float | None) -> float:
    if v is None:
        raise RuntimeError("solver returned no value")
    return float(v)


def optimise(
    prices: pd.Series, battery: Battery | None = None, sell_prices: pd.Series | None = None
) -> Schedule:
    """Maximise settle(...) subject to the battery's physical limits.

    `prices` are paid when charging and `sell_prices` (default: the same) earned when
    discharging. Different series let a risk-aware caller plan with a pessimistic buy price and
    a pessimistic sell price; settlement still happens at the actual price.

    A MILP rather than a pure LP: one binary per period forbids charging and discharging at
    once. Without it, at very negative prices the solver can cycle energy through the round-trip
    losses to import net energy and get paid for it, which a real battery can't do.
    """
    b = battery or Battery()
    idx = _validate(prices)
    sell_prices = prices if sell_prices is None else sell_prices
    if not sell_prices.index.equals(idx) or sell_prices.isna().any():
        raise ValueError("sell_prices must have the same index as prices and no NaN")
    buy = prices.to_numpy(dtype=float) / 1000  # £/MWh -> £/kWh
    sell = sell_prices.to_numpy(dtype=float) / 1000
    periods = range(len(buy))

    prob = pulp.LpProblem("battery", pulp.LpMaximize)
    # Bounds on the variables carry the power and SoC limits.
    charge = [prob.add_variable(f"charge_{t}", 0, b.power_kw) for t in periods]
    discharge = [prob.add_variable(f"discharge_{t}", 0, b.power_kw) for t in periods]
    soc = [prob.add_variable(f"soc_{t}", b.soc_min_kwh, b.capacity_kwh) for t in periods]
    charging = [prob.add_variable(f"charging_{t}", cat="Binary") for t in periods]

    # Objective: the same £ formula as settle(), term by term.
    prob += pulp.lpSum(
        (sell[t] * discharge[t] - buy[t] * charge[t]) * DT_H
        - b.degradation_gbp_per_kwh * (charge[t] + discharge[t]) * DT_H
        for t in periods
    )

    for t in periods:
        prev = b.initial_soc_kwh if t == 0 else soc[t - 1]
        prob += (
            soc[t] == prev + b.eta * charge[t] * DT_H - discharge[t] * DT_H / b.eta,
            f"energy_balance_{t}",
        )
        # charging=1 allows charge only; charging=0 allows discharge only.
        prob += charge[t] <= b.power_kw * charging[t], f"charge_mode_{t}"
        prob += discharge[t] <= b.power_kw * (1 - charging[t]), f"discharge_mode_{t}"

    if b.hourly_blocks:
        # A half-hour starting at :30 must match the one before it (same UTC hour).
        for t in periods:
            if t > 0 and idx[t].minute == 30 and idx[t] - idx[t - 1] == pd.Timedelta(minutes=30):
                # (PuLP 4: `var == var` compares the objects; a difference builds a constraint.)
                prob += charge[t] - charge[t - 1] == 0, f"hour_block_charge_{t}"
                prob += discharge[t] - discharge[t - 1] == 0, f"hour_block_discharge_{t}"

    # Terminal SoC: without it the optimum is to sell the stored energy for free money today.
    prob += soc[-1] == b.terminal, "terminal_soc"

    stats = prob.solve(pulp.HiGHS(msg=False))
    status = pulp.LpSolveStatus(stats.status)
    if status != pulp.LpSolveStatus.Optimal:
        raise RuntimeError(f"battery optimisation not optimal: {status.name}")

    frame = pd.DataFrame(
        {
            "price_gbp_per_mwh": prices.to_numpy(dtype=float),
            "charge_kw": [_solved(v.value()) for v in charge],
            "discharge_kw": [_solved(v.value()) for v in discharge],
            "soc_kwh": [_solved(v.value()) for v in soc],
        },
        index=idx,
    )
    return Schedule(frame=frame, objective_gbp=_solved(pulp.value(prob.objective)))
