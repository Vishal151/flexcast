"""`flexcast <command>` entrypoints. One image, many commands."""

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

import pandas as pd
import typer

from flexcast.config import Settings
from flexcast.ingest import DATASETS, Ingester, default_range
from flexcast.optimise import Battery
from flexcast.optimise import optimise as run_optimiser
from flexcast.prices import PriceSource, load_day_prices
from flexcast.settlement import LONDON
from flexcast.sources.elexon import ElexonClient
from flexcast.sources.gas import GasClient
from flexcast.sources.neso import NesoClient
from flexcast.sources.octopus import OctopusClient
from flexcast.storage import Storage

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def ingest(
    start: Annotated[str | None, typer.Option(help="First London day, YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="Last London day (inclusive)")] = None,
    datasets: Annotated[str | None, typer.Option(help="Comma-separated subset")] = None,
) -> None:
    """Ingest upstream data for London days START..END (default: yesterday and today)."""
    settings = Settings()
    first, last = default_range(datetime.now(UTC))
    first = date.fromisoformat(start) if start else first
    last = date.fromisoformat(end) if end else last
    names = datasets.split(",") if datasets else list(DATASETS)
    if unknown := set(names) - DATASETS.keys():
        raise typer.BadParameter(f"unknown datasets: {sorted(unknown)}")
    ingester = Ingester(
        Storage(settings.data_uri),
        ElexonClient(settings.elexon_base_url),
        OctopusClient(settings.octopus_base_url),
        region=settings.agile_region,
        neso=NesoClient(settings.neso_base_url),
        gas=GasClient(settings.gas_base_url),
    )
    failed = ingester.run(first, last, names, concurrency=settings.http_concurrency)
    if failed:
        typer.echo(f"failed days ({len(failed)}): {', '.join(map(str, failed))}", err=True)
        raise typer.Exit(1)


@app.command()
def optimise(
    day: Annotated[str, typer.Option("--date", help="London settlement day, YYYY-MM-DD")],
    end: Annotated[
        str | None, typer.Option(help="Last day (inclusive) to sum over a range")
    ] = None,
    prices: Annotated[str, typer.Option(help="system | agile")] = "agile",
) -> None:
    """Optimise one battery on known prices. One day prints the schedule; a range prints £."""
    if prices not in ("system", "agile"):
        raise typer.BadParameter("prices must be 'system' or 'agile'")
    source: PriceSource = "system" if prices == "system" else "agile"
    storage = Storage(Settings().data_uri)
    first = date.fromisoformat(day)
    last = date.fromisoformat(end) if end else first
    battery = Battery()

    if first == last:
        s = run_optimiser(load_day_prices(storage, source, first), battery)
        f = s.frame.copy()
        f.index = pd.DatetimeIndex(f.index).tz_convert(LONDON).strftime("%H:%M")
        typer.echo(f.round(2).to_string())
        typer.echo(f"\n{first} {prices}: £{s.objective_gbp:.2f}")
        return

    values: dict[date, float] = {}
    skipped: list[date] = []
    for i in range((last - first).days + 1):
        d = first + timedelta(days=i)
        try:
            values[d] = run_optimiser(load_day_prices(storage, source, d), battery).objective_gbp
        except (FileNotFoundError, ValueError):
            skipped.append(d)
    v = pd.Series(values)
    typer.echo(
        f"{prices} {first}..{last}: {len(v)} days, £{v.sum():.2f} total, "
        f"£{v.mean():.3f}/day mean, £{v.mean() * 365:.0f}/year annualised"
    )
    typer.echo(f"best day {v.idxmax()} £{v.max():.2f}; worst day {v.idxmin()} £{v.min():.2f}")
    if skipped:
        typer.echo(f"skipped {len(skipped)} incomplete/missing days: {skipped[:5]}...", err=True)


@app.command()
def backtest(
    test_start: Annotated[
        str | None, typer.Option(help="Default: 364 days before test-end")
    ] = None,
    test_end: Annotated[str | None, typer.Option(help="Default: last day with prices")] = None,
    train_start: Annotated[str | None, typer.Option(help="Default: first day with prices")] = None,
    report_uri: Annotated[str, typer.Option(help="Where results.md/png go")] = "file://./docs",
    experiment: Annotated[
        str | None, typer.Option(help="Log this run to experiments/ under this name")
    ] = None,
    description: Annotated[str, typer.Option(help="One line for the experiment log")] = "",
    market: Annotated[str, typer.Option(help="imbalance | day_ahead")] = "imbalance",
    extra_features: Annotated[
        str, typer.Option(help="Candidate input groups to add, comma-separated: gas,availability")
    ] = "",
) -> None:
    """Rolling-origin backtest of all models; writes the report and the last LightGBM model."""
    if market not in ("imbalance", "day_ahead"):
        raise typer.BadParameter("market must be 'imbalance' or 'day_ahead'")
    from flexcast import report
    from flexcast.backtest import (
        BacktestConfig,
        ModelFactory,
        add_risk_aware,
        run,
        sensitivity,
        summarise,
    )
    from flexcast.features import ALL_COLUMNS, DAY_AHEAD_MODEL_FEATURES, EXTRA_FEATURES
    from flexcast.history import History
    from flexcast.models.base import save_model
    from flexcast.models.baselines import (
        NaiveDayAhead,
        NaiveLastDay,
        SeasonalNaive,
        SeasonalNaiveDayAhead,
    )
    from flexcast.models.lgbm import LGBMQuantile
    from flexcast.optimise import Battery as BatteryCfg

    groups = [g for g in extra_features.split(",") if g]
    if unknown := set(groups) - EXTRA_FEATURES.keys():
        raise typer.BadParameter(f"unknown feature groups: {sorted(unknown)}")
    features = list(DAY_AHEAD_MODEL_FEATURES if market == "day_ahead" else ALL_COLUMNS)
    features += [c for g in groups for c in EXTRA_FEATURES[g]]

    settings = Settings()
    storage = Storage(settings.data_uri)
    history = History.load(storage)
    days = history.prices["london_date"]
    last = date.fromisoformat(test_end) if test_end else days.max()
    cfg = BacktestConfig(
        test_start=date.fromisoformat(test_start) if test_start else last - timedelta(days=364),
        test_end=last,
        train_start=date.fromisoformat(train_start) if train_start else days.min(),
        market="day_ahead" if market == "day_ahead" else "imbalance",
        battery=BatteryCfg(hourly_blocks=market == "day_ahead"),
    )
    factories: dict[str, ModelFactory]
    if cfg.market == "day_ahead":
        factories = {
            "naive_day_ahead": NaiveDayAhead,
            "seasonal_naive_day_ahead": SeasonalNaiveDayAhead,
            "lgbm_quantile": lambda: LGBMQuantile(features=features),
        }
    else:
        factories = {
            "naive_last_day": NaiveLastDay,
            "seasonal_naive": SeasonalNaive,
            "lgbm_quantile": lambda: LGBMQuantile(features=features),
        }
    result = run(history, cfg, factories)
    base_strategy = "lgbm_quantile"
    if settings.calibration_window_days > 0:
        from flexcast.models.calibration import add_calibrated

        add_calibrated(
            result,
            cfg.battery,
            "lgbm_quantile",
            "lgbm_calibrated",
            settings.calibration_window_days,
            settings.calibration_hours_per_block,
            tuple(settings.calibration_adjust.split(",")),
        )
        base_strategy = "lgbm_calibrated"
    risk = add_risk_aware(
        result, cfg.battery, base_strategy, (0.0, 0.25, 0.5, 0.75, 1.0), "lgbm_risk_aware"
    )
    sens = sensitivity(result.forecasts, cfg.battery, (0.0, 0.01, 0.02, 0.04))
    summary = summarise(result.daily)

    storage.write_parquet(result.daily, f"backtest/{market}/daily.parquet")
    storage.write_parquet(result.forecasts, f"backtest/{market}/forecasts.parquet")
    lgbm_model = result.models["lgbm_quantile"]
    assert isinstance(lgbm_model, LGBMQuantile)
    train_end = result.train_window[1]
    save_model(
        storage,
        result.models["lgbm_quantile"],
        f"models/{market}/lgbm_quantile/train_end={train_end}",
        {
            "git_sha": settings.git_sha,
            "train_window": [str(d) for d in result.train_window],
            "features": lgbm_model.features,
            "backtest": summary.loc["lgbm_quantile"].round(4).to_dict(),
            "risk_alpha": risk.alpha,
            "backtest_risk_aware": summary.loc[risk.strategy].round(4).to_dict(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    report.write(Storage(report_uri), result, cfg, sens, risk)
    if experiment:
        from flexcast.experiments import log_experiment

        # Keep this experiment's outputs, so notebooks can compare runs later.
        storage.write_parquet(result.daily, f"backtest/experiments/{experiment}/daily.parquet")
        storage.write_parquet(
            result.forecasts, f"backtest/experiments/{experiment}/forecasts.parquet"
        )
        storage.write_parquet(
            result.validation_forecasts,
            f"backtest/experiments/{experiment}/validation_forecasts.parquet",
        )
        save_model(storage, lgbm_model, f"backtest/experiments/{experiment}/model", {})
        log_experiment(
            Storage("file://./experiments"),
            experiment,
            description,
            {
                "market": market,
                "lgbm": lgbm_model.config(),
                "risk_alpha": risk.alpha,
                "calibration": {
                    "window_days": settings.calibration_window_days,
                    "hours_per_block": settings.calibration_hours_per_block,
                    "adjust": settings.calibration_adjust,
                },
                "battery": vars(cfg.battery),
            },
            # What production runs: (calibrated) forecast + risk-aware alpha on top.
            {str(k): v for k, v in summary.loc[risk.strategy].round(4).items()},
            window=f"{cfg.test_start}..{cfg.test_end}",
            git_sha=settings.git_sha,
        )
    typer.echo(summary.round(2).to_string())


@app.command()
def train(
    as_of: Annotated[
        str | None, typer.Option(help="ISO instant; trains on days known by then (default now)")
    ] = None,
) -> None:
    """Train the forecast model on all history known at AS_OF and mark it LATEST."""
    from flexcast.history import History
    from flexcast.jobs import train as train_model

    settings = Settings()
    storage = Storage(settings.data_uri)
    when = datetime.fromisoformat(as_of) if as_of else datetime.now(UTC)
    if when.tzinfo is None:
        raise typer.BadParameter("--as-of must include a timezone, e.g. 2025-10-28T12:00:00+00:00")
    prefix = train_model(History.load(storage), storage, when, settings.git_sha, settings.market)
    typer.echo(f"trained {prefix}")


@app.command()
def plan(
    day: Annotated[
        str | None, typer.Option("--date", help="Target London day (default tomorrow)")
    ] = None,
) -> None:
    """Forecast and schedule one day using the LATEST model."""
    from flexcast.history import History
    from flexcast.jobs import plan as make_plan

    settings = Settings()
    storage = Storage(settings.data_uri)
    target = (
        date.fromisoformat(day)
        if day
        else datetime.now(UTC).astimezone(LONDON).date() + timedelta(days=1)
    )
    result = make_plan(History.load(storage), storage, target, settings)
    typer.echo(f"planned {target}: expected £{result.expected_gbp:.2f}")


@app.command()
def serve() -> None:
    """Run the HTTP API on $PORT (default 8080)."""
    import uvicorn

    from flexcast.api import create_app

    uvicorn.run(create_app(), host="0.0.0.0", port=Settings().port, log_level="info")  # noqa: S104
