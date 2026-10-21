"""Headline chart for the README, from the logged experiments (production strategies).

uv run python scripts/readme_chart.py    # writes docs/headline.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import style  # type: ignore[import-not-found]  # noqa: E402

from flexcast.backtest import summarise  # noqa: E402
from flexcast.storage import Storage  # noqa: E402

RUNS = {  # market -> (experiment, naive strategy)
    "Imbalance price": ("calibrated", "naive_last_day"),
    "Day-ahead auction": ("day_ahead", "naive_day_ahead"),
}
MODEL = "lgbm_risk_aware"  # the production row (calibrated forecast + validation-chosen alpha)


def main() -> None:
    style.use()
    store = Storage(f"file://{ROOT / 'data'}")
    daily = {
        m: store.read_parquet(f"backtest/experiments/{e}/daily.parquet")
        for m, (e, _) in RUNS.items()
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[1.5, 1])

    wide = daily["Day-ahead auction"].pivot(
        index="target_date", columns="strategy", values="value_gbp"
    )
    for col, color, label, ls in (
        ("perfect", style.TEXT_2, "perfect foresight", "--"),
        (MODEL, style.BLUE, "our forecast", "-"),
        ("naive_day_ahead", style.ORANGE, "copy yesterday's prices", "-"),
    ):
        cum = wide[col].cumsum()
        left.plot(wide.index, cum, color=color, linestyle=ls, label=label)
        left.annotate(
            f"£{cum.iloc[-1]:.0f}",
            (wide.index[-1], cum.iloc[-1]),
            xytext=(4, -3),
            textcoords="offset points",
            color=color,
            fontsize=10,
        )
    left.set(title="Day-ahead auction: £ earned per home battery, test year", ylabel="£")
    left.axhline(0, color=style.TEXT_2, linewidth=0.8)
    left.legend(loc="upper left", frameon=False)
    left.tick_params(axis="x", labelrotation=30)

    labels, model_pct, naive_pct = [], [], []
    for m, (_, naive) in RUNS.items():
        s = summarise(daily[m])
        labels.append(m)
        model_pct.append(s.loc[MODEL, "pct_of_perfect"])
        naive_pct.append(s.loc[naive, "pct_of_perfect"])
    x = range(len(labels))
    w = 0.38
    bars_m = right.bar([i - w / 2 for i in x], model_pct, w, color=style.BLUE, label="our forecast")
    bars_n = right.bar(
        [i + w / 2 for i in x], naive_pct, w, color=style.ORANGE, label="copy yesterday"
    )
    for bars in (bars_m, bars_n):
        right.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=10)
    right.axhline(0, color=style.TEXT_2, linewidth=0.8)
    right.set_xticks(list(x), labels)
    right.tick_params(axis="x", length=0)
    right.set(title="Share of perfect-foresight value captured", ylabel="%", ylim=(-75, 85))
    right.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    out = ROOT / "docs" / "headline.png"
    fig.savefig(out, dpi=130, facecolor=style.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
