# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 · Making the uncertainty band honest (calibration)
#
# Notebook 02: the p10–p90 band holds ~66% of outcomes instead of 80%, and prices land above
# p90 about twice as often as they should, worst in the evening. A forecast that is wrong in a
# **consistent** way can be corrected after the fact.
#
# **Method: rolling conformal calibration.** For each target day, look at the model's own
# forecasts over the previous N days whose prices are already known, and measure how far off each
# quantile was. Shift p90 by the 90th percentile of (actual − p90), p10 by the 10th percentile of
# (actual − p10), and so on. Optionally do this per 4-hour block of the day. Because it only uses
# days known at decision time, it runs identically in the backtest and in the daily plan job.
#
# Choices compared on the validation window: window length (30/60/90 days), one adjustment for the
# day vs per 4-hour block, and band only (p10/p90) vs band and p50. Adjusting p50 moves the
# battery's decisions; adjusting only the band changes the stated uncertainty, not the schedule.

# %%
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import style

from flexcast.backtest import BacktestConfig, run, summarise, valuate
from flexcast.history import History
from flexcast.models.calibration import rolling_calibration
from flexcast.models.lgbm import LGBMQuantile
from flexcast.optimise import Battery
from flexcast.settlement import LONDON
from flexcast.storage import Storage

style.use()
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
history = History.load(Storage(f"file://{ROOT / 'data'}"))
VAL_START, VAL_END, TRAIN_START = date(2025, 4, 21), date(2025, 10, 19), date(2024, 10, 20)
HISTORY_START = date(2025, 1, 20)  # 90 days of forecasts before the validation window

# The current model (defaults reflect notebook 04's choice), forecasting with weekly retraining.
cfg = BacktestConfig(
    test_start=HISTORY_START, test_end=VAL_END, train_start=TRAIN_START, validation_days=0
)
base = run(history, cfg, {"model": LGBMQuantile}).forecasts
print(f"forecast days: {base.target_date.nunique()} ({HISTORY_START} to {VAL_END})")

# %% [markdown]
# ## Compare calibration choices (scored on the validation window only)

# %%
variants = {}
f = base
for window in (30, 60, 90):
    for blocks, block_label in ((None, "whole day"), (4, "4h blocks")):
        for adjust, adjust_label in ((("p10", "p90"), "band"), (("p10", "p50", "p90"), "band+p50")):
            name = f"{window}d · {block_label} · {adjust_label}"
            f = rolling_calibration(f, "model", name, window, blocks, adjust)
            variants[name] = (window, block_label, adjust_label)

val = f[f["target_date"] >= VAL_START]
daily = valuate(val, Battery())
s = summarise(daily).drop(index="perfect")
below = val.assign(b=val.actual < val.p10).groupby("strategy")["b"].mean()
above = val.assign(a=val.actual > val.p90).groupby("strategy")["a"].mean()
s = s.assign(below_p10=below, above_p90=above).sort_values("pinball")
s[["coverage_80", "below_p10", "above_p90", "pinball", "mae", "gbp_per_year"]].style.format(
    {
        "coverage_80": "{:.0%}",
        "below_p10": "{:.0%}",
        "above_p90": "{:.0%}",
        "pinball": "{:.3f}",
        "mae": "{:.2f}",
        "gbp_per_year": "{:.1f}",
    }
)

# %% [markdown]
# ## Reliability: does each quantile mean what it says?
#
# For a well-calibrated forecast, the actual price falls below p10 10% of the time, below p50 50%
# and below p90 90%. Points on the diagonal are honest.

# %%
rule_ok = s[(s.coverage_80 - 0.8).abs() <= 0.03]
chosen = (rule_ok if len(rule_ok) else s).index[0]


def reliability(frame: pd.DataFrame) -> list[float]:
    return [(frame.actual < frame[q]).mean() for q in ("p10", "p50", "p90")]


fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], color=style.TEXT_2, linewidth=1, linestyle="--")
for name, color, label in (
    ("model", style.ORANGE, "uncalibrated"),
    (chosen, style.BLUE, "calibrated"),
):
    r = reliability(val[val.strategy == name])
    ax.plot([0.1, 0.5, 0.9], r, color=color, marker="o", markersize=8, label=label)
ax.set(
    title="Reliability on the validation window",
    xlabel="nominal quantile",
    ylabel="share of actuals below it",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax.legend(loc="upper left")
plt.show()

# %%
by_hour = val[val.strategy.isin(["model", chosen])].copy()
by_hour["hour"] = by_hour["start_time"].dt.tz_convert(LONDON).dt.hour
cov = (
    by_hour.assign(inside=by_hour.actual.between(by_hour.p10, by_hour.p90)).pivot_table(
        index="hour", columns="strategy", values="inside"
    )
    * 100
)
fig, ax = plt.subplots()
for name, color, label in (
    ("model", style.ORANGE, "uncalibrated"),
    (chosen, style.BLUE, "calibrated"),
):
    ax.plot(cov.index, cov[name], color=color, label=label)
ax.axhline(80, color=style.TEXT_2, linewidth=1, linestyle="--")
ax.set(
    title="Share of actuals inside the p10–p90 band, by hour",
    xlabel="London hour",
    ylabel="% inside band",
    ylim=(0, 100),
)
ax.legend(loc="lower left")
plt.show()

# %% [markdown]
# ## Decision
#
# Rule, set before running: among variants whose validation coverage is within 80% ± 3 points,
# pick the lowest pinball. Report £ alongside: a band-only calibration can't change £ (the
# schedule is planned on p50), while band+p50 can.

# %%
print(f"chosen: {chosen}  →  window, blocks, adjust = {variants[chosen]}")
print(
    s.loc[
        [chosen, "model"], ["coverage_80", "below_p10", "above_p90", "pinball", "gbp_per_year"]
    ].round(3)
)

# %% [markdown]
# ### Reading the result
#
# - **Adopted: 90-day window, one offset for the whole day, band only** (now the code default).
#   Validation coverage 72% → 82%, misses balanced (9% below p10, 9% above p90, vs 12% and 16%),
#   pinball 8.12 → 8.08. £ is unchanged by construction: the schedule is planned on p50.
# - **The simplest version won.** Per-4-hour-block offsets and shifting p50 didn't improve pinball,
#   and the block versions that moved p50 cost £: separate shifts per block distort the daily shape
#   the battery trades on.
# - **Why fit it on past forecasts, not the test year:** calibration *is* a fitted model of the
#   forecast's errors. Fitting it on the days you then score would make coverage look perfect by
#   construction, which is the same leakage as training on the test set.
# - **What it buys:** an honest p10/p90 is what a risk-aware schedule (Stage 2b) or a trader needs
#   to size risk. With the band now trustworthy, the backtest re-chooses the risk-aware α on top of
#   the calibrated forecast.
