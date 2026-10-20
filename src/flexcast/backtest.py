"""Rolling-origin backtest and £ valuation.

For each target day D in the test window: features as of D-1 09:00 London; models retrained every
`retrain_every` days on target days whose prices were all known by then; forecast; then
optimise the battery on each forecast's p50 and settle at actual prices. "perfect" is the
forecast that equals the actuals (perfect foresight: the upper bound).

Forecasting and valuation are separate steps, so valuation can be re-run (e.g. for a different
degradation cost) without retraining.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta

import pandas as pd

from flexcast.features import build, decision_time_for
from flexcast.history import PRICE_LAG, History
from flexcast.models.base import COLUMNS, Model, pinball_loss
from flexcast.optimise import Battery, optimise, settle
from flexcast.settlement import london_day_bounds, period_starts

log = logging.getLogger(__name__)

PERFECT = "perfect"
ModelFactory = Callable[[], Model]


@dataclass(frozen=True)
class BacktestConfig:
    test_start: date
    test_end: date  # inclusive
    train_start: date
    retrain_every_days: int = 7
    decision_at: time = time(9)
    battery: Battery = field(default_factory=Battery)
    # Days before test_start that are forecast the same way and used only to choose settings
    # (e.g. the risk-aware alpha), so nothing is tuned on the test window.
    validation_days: int = 182

    @property
    def forecast_start(self) -> date:
        return self.test_start - timedelta(days=self.validation_days)


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame  # test window: target_date, start_time, strategy, p10..p90, actual
    validation_forecasts: pd.DataFrame  # same columns, validation window
    daily: pd.DataFrame  # test window: target_date, strategy, value_gbp, mae, pinball
    models: dict[str, Model]  # as last trained
    train_window: tuple[date, date]  # of the last retrain
    skipped: list[date]


def days_between(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def training_cutoff(decision_time: datetime) -> date:
    """Last target day whose every price was available before decision_time."""
    d = decision_time.date()
    while True:
        _, end = london_day_bounds(d)
        last_available = end - timedelta(minutes=30) + PRICE_LAG
        if last_available < decision_time:
            return d
        d -= timedelta(days=1)


def feature_table(
    history: History, days: list[date], decision_at: time
) -> tuple[pd.DataFrame, pd.Series, list[date]]:
    """Features and actuals for every day with a complete set of actual prices.

    Each day's features depend only on its own decision time, so one table serves every retrain.
    """
    xs, ys, skipped = [], [], []
    for d in days:
        actual = history.actual_prices(d)
        if len(actual) != len(period_starts(d)):
            skipped.append(d)
            continue
        X = build(history, decision_time_for(d, decision_at), d).X
        xs.append(X.assign(target_date=d).set_index("target_date", append=True))
        ys.append(
            actual.rename("price")
            .to_frame()
            .assign(target_date=d)
            .set_index("target_date", append=True)["price"]
        )
    if not xs:
        return pd.DataFrame(), pd.Series(dtype=float), skipped
    X_all = pd.concat(xs).reorder_levels(["target_date", "start_time"])
    y_all = pd.concat(ys).reorder_levels(["target_date", "start_time"])
    return X_all, y_all, skipped


def run(
    history: History, cfg: BacktestConfig, factories: dict[str, ModelFactory]
) -> BacktestResult:
    X_all, y_all, skipped = feature_table(
        history, days_between(cfg.train_start, cfg.test_end), cfg.decision_at
    )
    target_dates = X_all.index.get_level_values("target_date")
    models: dict[str, Model] = {}
    last_fit: date | None = None
    train_window = (cfg.train_start, cfg.train_start)
    rows: list[pd.DataFrame] = []

    for d in days_between(cfg.forecast_start, cfg.test_end):
        decision_time = decision_time_for(d, cfg.decision_at)
        if last_fit is None or (d - last_fit).days >= cfg.retrain_every_days:
            cutoff = training_cutoff(decision_time)
            mask = (target_dates >= cfg.train_start) & (target_dates <= cutoff)
            for name, factory in factories.items():
                models[name] = factory()
                models[name].fit(X_all[mask], y_all[mask])
            last_fit, train_window = d, (cfg.train_start, cutoff)
            log.info("retrained on %s..%s (%d rows) for %s", cfg.train_start, cutoff, mask.sum(), d)
        if d not in target_dates:
            continue
        on_day = target_dates == d
        X = X_all[on_day].droplevel("target_date")
        actual = y_all[on_day].droplevel("target_date")
        preds = {PERFECT: pd.DataFrame({c: actual for c in COLUMNS})}
        preds |= {name: m.predict(X) for name, m in models.items()}
        for name, p in preds.items():
            rows.append(p.assign(actual=actual, strategy=name, target_date=d).reset_index())

    every = pd.concat(rows, ignore_index=True)
    is_test = every["target_date"] >= cfg.test_start
    forecasts = every[is_test].reset_index(drop=True)
    return BacktestResult(
        forecasts=forecasts,
        validation_forecasts=every[~is_test].reset_index(drop=True),
        daily=valuate(forecasts, cfg.battery),
        models=models,
        train_window=train_window,
        skipped=[s for s in skipped if s >= cfg.test_start],
    )


def valuate(forecasts: pd.DataFrame, battery: Battery) -> pd.DataFrame:
    """Optimise each (day, strategy) and settle at actuals.

    The schedule is planned on `buy`/`sell` columns where present (risk-aware rows), else p50.
    """
    out = []
    for (d, strategy), g in forecasts.groupby(["target_date", "strategy"], sort=True):
        g = g.set_index("start_time")
        buy = g["buy"] if "buy" in g and g["buy"].notna().all() else g["p50"]
        sell = g["sell"] if "sell" in g and g["sell"].notna().all() else g["p50"]
        schedule = optimise(buy, battery, sell_prices=sell)
        out.append(
            {
                "target_date": d,
                "strategy": strategy,
                "value_gbp": settle(schedule.frame, g["actual"], battery),
                "mae": float((g["p50"] - g["actual"]).abs().mean()),
                "pinball": pinball_loss(g["actual"], g[COLUMNS]),
                "coverage_80": float(g["actual"].between(g["p10"], g["p90"]).mean()),
            }
        )
    return pd.DataFrame(out)


def summarise(daily: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: days, £/battery/year, % of perfect foresight captured, MAE, pinball."""
    pf = daily[daily["strategy"] == PERFECT].set_index("target_date")["value_gbp"]
    rows = []
    for strategy, g in daily.groupby("strategy"):
        g = g.set_index("target_date")
        rows.append(
            {
                "strategy": strategy,
                "days": len(g),
                "gbp_per_year": g["value_gbp"].mean() * 365,
                "pct_of_perfect": 100 * g["value_gbp"].sum() / pf.reindex(g.index).sum(),
                "mae": g["mae"].mean(),
                "pinball": g["pinball"].mean(),
                "coverage_80": g["coverage_80"].mean(),
            }
        )
    out = pd.DataFrame(rows).set_index("strategy")
    return out.sort_values("gbp_per_year", ascending=False)


def worst_days(daily: pd.DataFrame, strategy: str, n: int = 10) -> pd.DataFrame:
    """Days where `strategy` fell furthest short of perfect foresight, in £."""
    wide = daily.pivot(index="target_date", columns="strategy", values="value_gbp")
    out = pd.DataFrame(
        {
            "perfect_gbp": wide[PERFECT],
            f"{strategy}_gbp": wide[strategy],
            "shortfall_gbp": wide[PERFECT] - wide[strategy],
        }
    )
    return out.nlargest(n, "shortfall_gbp")


def sensitivity(
    forecasts: pd.DataFrame, battery: Battery, costs_gbp_per_kwh: tuple[float, ...]
) -> pd.DataFrame:
    """£/battery/year and % of perfect per strategy, re-valued at each degradation cost."""
    cols = {}
    for cost in costs_gbp_per_kwh:
        s = summarise(valuate(forecasts, replace(battery, degradation_gbp_per_kwh=cost)))
        cols[f"{cost * 100:g}p/kWh"] = [
            f"{'-' if v < 0 else ''}£{abs(v):.0f} ({p:.0f}%)"
            for v, p in zip(s["gbp_per_year"], s["pct_of_perfect"], strict=True)
        ]
        index = s.index
    return pd.DataFrame(cols, index=index)


def with_risk_aware(
    forecasts: pd.DataFrame, strategy: str, alpha: float, name: str
) -> pd.DataFrame:
    """Append a risk-aware copy of `strategy`: plan buying at p50 + alpha*(p90-p50) and selling
    at p50 - alpha*(p50-p10). alpha=0 is the p50 strategy; alpha=1 plans at p90/p10. A trade
    then happens only if it pays even when prices land at the pessimistic end of the range."""
    base = forecasts[forecasts["strategy"] == strategy]
    risk = base.assign(
        strategy=name,
        buy=base["p50"] + alpha * (base["p90"] - base["p50"]),
        sell=base["p50"] - alpha * (base["p50"] - base["p10"]),
    )
    return pd.concat([forecasts, risk], ignore_index=True)


def alpha_grid(
    forecasts: pd.DataFrame, battery: Battery, strategy: str, alphas: tuple[float, ...]
) -> pd.DataFrame:
    """£/battery/year and % of perfect for `strategy` at each alpha."""
    rows = {}
    keep = forecasts["strategy"].isin([strategy, PERFECT])
    for a in alphas:
        name = f"alpha={a:g}"
        daily = valuate(with_risk_aware(forecasts[keep], strategy, a, name), battery)
        s = summarise(daily).loc[name]
        rows[name] = {"gbp_per_year": s["gbp_per_year"], "pct_of_perfect": s["pct_of_perfect"]}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("alpha")


def select_alpha(
    validation_forecasts: pd.DataFrame,
    battery: Battery,
    strategy: str,
    alphas: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    """The alpha with the most £ on the validation window (ties -> smallest alpha)."""
    grid = alpha_grid(validation_forecasts, battery, strategy, alphas)
    best = grid["gbp_per_year"].to_numpy().argmax()
    return alphas[best], grid


@dataclass(frozen=True)
class RiskAware:
    strategy: str  # the new strategy's name
    base: str  # the strategy it adjusts
    alpha: float  # chosen on the validation window
    validation_grid: pd.DataFrame
    test_grid: pd.DataFrame  # hindsight only; never used for the choice


def add_risk_aware(
    result: BacktestResult,
    battery: Battery,
    base: str,
    alphas: tuple[float, ...],
    name: str,
) -> RiskAware:
    """Choose alpha on validation days, then add `name` to the test results at that alpha."""
    alpha, val_grid = select_alpha(result.validation_forecasts, battery, base, alphas)
    test_grid = alpha_grid(result.forecasts, battery, base, alphas)
    result.forecasts = with_risk_aware(result.forecasts, base, alpha, name)
    risk_rows = result.forecasts[result.forecasts["strategy"] == name]
    result.daily = pd.concat([result.daily, valuate(risk_rows, battery)], ignore_index=True)
    return RiskAware(name, base, alpha, val_grid, test_grid)
