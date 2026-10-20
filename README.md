# flexcast

Forecast-driven scheduling for a home battery on Great Britain's electricity market: predict
tomorrow's half-hourly prices, plan when to charge and discharge, and measure the value in £
against perfect foresight and simple baselines.

**Status:** work in progress, built in stages (see [SPEC.md](SPEC.md) and
[docs/journal.md](docs/journal.md)). Current backtest results: [docs/results.md](docs/results.md).
The full write-up comes at the end.

## Run it

Needs Docker and [uv](https://docs.astral.sh/uv/).

```sh
make lint test        # static checks + unit tests (no network, no Docker)
make up && make itest # local stack with emulated cloud storage + integration tests
make demo             # load sample data, train, plan; then open http://localhost:8080/docs
make down
```

## Data

Contains BMRS data © Elexon Limited copyright and database right.
Tariff data from the [Octopus Energy public API](https://developer.octopus.energy/).
