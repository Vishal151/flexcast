# flexcast — Spec

A forecast-driven scheduler for flexible home energy assets (battery / EV) on GB electricity market data. Ingests public market data, forecasts half-hourly prices, optimises a dispatch schedule, and reports the value **in £**. Runs on GCP in Docker.

Built as a portfolio and learning project for a full-stack data science role in energy flexibility. Optimise for: correctness, reproducibility, a clear commercial story, and **learning each layer properly**. Not for: scale, UI polish, framework count.

**Principle: local first.** Everything runs and is tested on a laptop — first as plain Python, then in Docker against GCP emulators — before anything touches a real GCP project. Cloud is the last stage, not the first.

---

## 1. Problem statement

An aggregator controls a fleet of home batteries. Each day, before a decision deadline, it must commit a half-hourly charge/discharge schedule for the next day. Prices for that day are not yet known.

The project answers three questions:
1. How much is a battery worth with **perfect foresight** of prices? (upper bound)
2. How much does it earn using **our forecast**? (what we can actually capture)
3. How much does it earn using a **naive forecast** (yesterday's prices)? (the bar to beat)

**Headline metric:** % of perfect-foresight value captured, vs naive. Forecast accuracy (MAE, pinball loss) is secondary — it only matters insofar as it moves £.

### Simplifications (state these in the README)
- Battery is a price-taker settled at the half-hourly system (imbalance) price. A real home battery can't access that price directly — only via an aggregator. That gap is the aggregator's business model; say so.
- One representative battery, scaled linearly to fleet size for reporting.
- No network charges; degradation is a simple £/kWh throughput cost.

---

## 2. Data sources

All public, free, no key. **Stage 0 must verify every path below against the live OpenAPI spec** — do not trust this list blindly.

| Source | What | Notes |
|---|---|---|
| Elexon Insights API `https://data.elexon.co.uk/bmrs/api/v1` | System prices (SSP/SBP) — `/balancing/settlement/system-prices/{settlementDate}` | Forecast target. Store both prices. |
| Elexon dataset `MID` | Market Index price | Feature (lagged) and alternative target. Use provider `APXMIDP`; `N2EXMIDP` has missing periods |
| Elexon dataset `NDF` | Day-ahead national demand forecast | Feature. Must use the vintage published **before** decision time |
| Elexon dataset `WINDFOR` | Wind generation forecast | Feature. Same vintage rule. **Hourly** resolution: upsample to half-hourly in features |
| Elexon dataset `FUELHH` | Half-hourly generation by fuel | Feature (lagged) / EDA |
| Elexon dataset `TEMP` | Temperature | Feature (optional, low priority). **Daily** resolution, one row per day |
| Octopus API `https://api.octopus.energy/v1/products/` | Agile half-hourly unit rates | Stage 1 only — known prices to validate optimiser. Discover current Agile product code via `/products`; don't hardcode. |

OpenAPI definitions: https://developer.data.elexon.co.uk/ — dataset endpoints support querying by **publish time**, which is how we get point-in-time forecast vintages.

Attribution required in README: "Contains BMRS data © Elexon Limited copyright and database right".

### Known gotchas
- **Settlement periods**: 48 per day normally, **46** on spring DST day, **50** on autumn DST day. Store timestamps in UTC; derive settlement date/period, never the reverse by arithmetic.
- **Agile day-ahead rates** publish ~16:00 London; before then today's query returns <48 rows.
- **Forecast vintages**: for each target day, keep only forecasts with `publishTime < decision_time`. Using the latest revision is leakage.
- **Query window cap**: `NDF`/`WINDFOR` publish-time queries are capped at 1 day per request; clients must chunk.
- **MID `from`/`to` are instants**, not dates: query with the London-day UTC bounds.
- Prices can be **negative** and **spiky**. Don't clip in storage; decide clipping in modelling, explicitly.

---

## 3. Architecture

### Target (GCP)
```
Cloud Scheduler
  ├─ ingest job (Cloud Run Job, every 30 min)  → raw Parquet in GCS
  ├─ train job  (Cloud Run Job, weekly)        → model artefact + metadata JSON in GCS
  └─ plan job   (Cloud Run Job, daily, D-1)    → forecast + schedule Parquet in GCS
Cloud Run service (FastAPI): /schedule /forecast /health /metrics
[Stage 6, optional] Pub/Sub: simulated fleet telemetry → consumer → GCS
```

### Local (docker compose) — same image, same code, emulated dependencies
```
docker compose
  ├─ gcs           fsouza/fake-gcs-server        (GCS emulator)
  ├─ upstream-mock small FastAPI app serving recorded Elexon/Octopus fixtures
  ├─ api           flexcast serve
  ├─ ingest/train/plan   flexcast <cmd>  (run on demand: `docker compose run --rm ingest`)
  └─ pubsub        Pub/Sub emulator (Stage 6 only)
```

### Environment profiles
One code path; behaviour changes only through env vars.

| Profile | `FLEXCAST_DATA_URI` | Upstream APIs | Emulator env |
|---|---|---|---|
| `local` (Stages 0–2) | `file://./data` | live (or fixtures in tests) | — |
| `compose` (Stages 3–4) | `gs://flexcast-local` | `upstream-mock` | `STORAGE_EMULATOR_HOST=http://gcs:4443` |
| `gcp` (Stage 5) | `gs://flexcast-<env>` | live | — (real ADC credentials) |

Storage goes through **fsspec** (`file://` and `gs://` via `gcsfs`), so switching profile is a URI change, not a code change. `gcsfs` honours `STORAGE_EMULATOR_HOST`, so the compose profile exercises the real GCS client against the emulator.

### What can't be emulated locally (test these in Stage 5)
- **Cloud Scheduler** — no emulator. Locally, schedules are config; jobs are triggered by `make`/`docker compose run`. The cron expressions are deployed in Stage 5.
- **IAM / service accounts** — only real in GCP. Keep to one least-privilege SA.
- **Cloud Run runtime specifics** — `PORT` env, request timeouts, memory/CPU limits. Mirror them in compose (`PORT`, `mem_limit`) so surprises are small.

### Design choices (and why)
- **Parquet + DuckDB, not a database.** Data is small (~17.5k rows/year per series). Zero idle cost. No BigQuery emulator needed.
- **Cloud Run Jobs, not Airflow/Kubernetes.** Three scheduled batch jobs don't need an orchestrator.
- **joblib artefacts + metadata JSON, not MLflow.** Metadata: git SHA, training window, features, backtest metrics, created_at.
- **One Docker image, multiple entrypoints** (`flexcast ingest|train|plan|serve|consume`).

### Package layout
```
src/flexcast/
  config.py        # pydantic-settings; profiles via env
  storage.py       # fsspec-backed read/write of Parquet + JSON
  sources/         # elexon.py, octopus.py — thin httpx clients; base URLs from config
  ingest.py        # pull → validate → write partitions; idempotent
  features.py      # point-in-time feature build for a decision_time
  models/          # baselines.py, lgbm.py — common fit/predict interface
  optimise.py      # battery LP
  backtest.py      # rolling-origin backtest + £ valuation
  api.py           # FastAPI
  telemetry.py     # Stage 6: publisher (simulated fleet) + consumer
  cli.py           # typer entrypoints
mock_upstream/     # FastAPI stub serving tests/fixtures/*
tests/
  unit/            # no network, no Docker, < 30 s total
  integration/     # needs `docker compose up`; marker: integration
  live/            # hits real Elexon for one day; marker: live; opt-in
deploy/            # gcloud scripts (Terraform as stretch)
compose.yaml
```

---

## 4. Testing strategy

| Layer | Runs where | What it proves | Command |
|---|---|---|---|
| **Unit** | plain Python | Logic: DST helper, LP constraints, leakage, feature build, valuation maths | `make test` |
| **Integration** | compose stack | Real HTTP clients → mock upstream; real GCS client → emulator; container entrypoints; API contract | `make itest` |
| **Live smoke** | laptop/CI, opt-in | Upstream APIs haven't changed shape | `make live` |
| **Deployed smoke** | against GCP dev | IAM, Scheduler, Cloud Run config | `make smoke ENV=dev` |

Rules:
- Unit tests never need Docker or network. Record fixtures once (`scripts/record_fixtures.py`), replay with `respx`.
- Integration tests create a fresh bucket per test session in the emulator and tear it down.
- The same fixtures feed both `respx` (unit) and `upstream-mock` (integration) — one source of truth.

---

## 5. Stages

Each stage has a **learning goal**, a **build** list, **acceptance** criteria, and **check yourself** questions. Each ends with passing tests, a commit, and 3–5 lines in `docs/journal.md`. Do stages in order; don't start the next until the current one's acceptance passes.

### Stage 0 — Explore the data (½ day)
**Learn:** what GB market data looks like; settlement periods; forecast vintages.
**Build:**
- `scripts/explore_api.py`: hit each §2 source for one normal day and one DST day; print shapes, columns, dtypes, row counts, publish times.
- `scripts/record_fixtures.py`: save responses to `tests/fixtures/`.
- `docs/data-sources.md`: verified endpoints, params, response fields, publish cadence, quirks.
**Accept:** doc exists; every §2 source verified or replaced with what was used instead; fixtures committed.
**Check yourself:** How many NDF vintages exist for a given target day, and when is each published? What's the latest one you could have seen at D-1 09:00?

### Stage 1 — Core library, local only (1 day)
**Learn:** package structure with uv; httpx clients; idempotent ingestion; LP formulation.
**Build:**
- `config.py` with the `local` profile; `storage.py` over fsspec (`file://` only for now).
- Elexon + Octopus clients (retry/backoff, chunked ranges, base URL from config).
- `flexcast ingest --start --end`; no args = last 2 days. Dedupe on natural key + publish time. Schema validation (pandera or pydantic); failures logged, not dropped.
- Backfill ≥ 2 years into `./data`.
- Battery LP (cvxpy + HiGHS, or PuLP): capacity, power limits, round-trip efficiency, initial/terminal SoC, degradation £/kWh. Decide LP vs MILP for negative-price simultaneous charge/discharge; test the choice.
- Validate optimiser against Agile (known prices).
**Accept:** re-running ingest gives identical files; DST days have 46/50 periods; LP unit tests (SoC bounds, energy balance, terminal SoC, flat price → no cycling, hand-solvable two-price case).
**Check yourself:** Why is the LP's terminal SoC constraint necessary? What happens to arbitrage value if round-trip efficiency drops from 90% to 80%?

### Stage 2 — Forecasting and backtest, local only (1½ days)
**Learn:** point-in-time features; leakage; quantile models; valuing a forecast in £.
**Build:**
- `features.py::build(decision_time, target_date)` — only data published before `decision_time` (default D-1 09:00 Europe/London, configurable).
- Features: calendar (period, dow, holiday, DST flag), lagged system/MID prices, NDF/WINDFOR vintages for target day, rolling stats.
- Models behind one interface: `NaiveYesterday`, `SeasonalNaive` (same period last week), `LGBMQuantile` (p10/p50/p90).
- **Leakage test (required):** every feature's source publish time < decision_time.
- Rolling-origin backtest ≥ 12 months: retrain weekly, forecast daily. Each day: optimise on forecast → settle at actuals. Same for perfect foresight and baselines.
- Output: £/battery/year, % of perfect foresight captured, MAE, pinball loss, worst 10 days; fleet scaling for 1k/10k/100k batteries (price-impact caveat). Results → `docs/results.md` with one chart.
- Stretch: does a risk-aware schedule (p10/p90) beat p50 in £?
**Accept:** leakage test passes; models round-trip through serialisation; `flexcast backtest` reproduces the table from a clean checkout.
**Check yourself:** Which feature, if accidentally taken from the latest vintage, would inflate results most? Why might a lower-MAE model capture less £?

### Stage 3 — Containerise and emulate GCP locally (1 day)
**Learn:** Docker multi-stage builds; compose; env-driven config; emulators; integration testing.
**Build:**
- Multi-stage Dockerfile (uv, slim base, non-root, `PORT` honoured). Image < 500 MB.
- `compose.yaml` with `gcs`, `upstream-mock`, `api`, and job services (§3). Healthchecks + `depends_on: condition: service_healthy`.
- `compose` profile: `FLEXCAST_DATA_URI=gs://flexcast-local`, `STORAGE_EMULATOR_HOST`, upstream base URLs → `upstream-mock`.
- `mock_upstream/`: serves the Stage 0 fixtures at the same paths as the real APIs.
- FastAPI endpoints: `/schedule?date=`, `/forecast?date=`, `/health`, `/metrics` (rolling 30-day forecast error and £ capture).
- Integration tests (`-m integration`): ingest → train → plan end-to-end in containers; files land in emulated GCS; API returns a valid schedule; re-running ingest is idempotent through the real GCS client.
- `make up`, `make down`, `make itest`.
**Accept:** `make up && make itest` passes on a clean machine with only Docker installed; no code differences between `local` and `compose` profiles beyond config.
**Check yourself:** What would break if the app wrote to `/app/data` instead of going through `FLEXCAST_DATA_URI`? Why does the emulator need `STORAGE_EMULATOR_HOST` rather than a different code path?

### Stage 4 — CI (½ day)
**Learn:** GitHub Actions; running compose in CI; caching.
**Build:**
- Workflow on PR: ruff, mypy, unit tests, `docker build`, `docker compose up` + integration tests.
- Weekly scheduled workflow: live smoke test (`-m live`) — catches upstream API drift.
- Cache uv and Docker layers.
**Accept:** green PR checks; integration job < 10 min.
**Check yourself:** Why keep live tests out of PR checks?

### Stage 5 — Deploy to GCP (1 day)
**Learn:** Artifact Registry, Cloud Run Jobs/services, Cloud Scheduler, IAM, cost.
**Build:**
- One GCP project (`dev`). Budget alert at £10.
- `deploy/` gcloud scripts (Terraform stretch): Artifact Registry, GCS bucket, 3 Cloud Run Jobs, 1 Cloud Run service, Scheduler triggers, one least-privilege service account.
- Seed GCS with the backfill from Stage 1 (`gcloud storage cp`) rather than re-pulling.
- Deployed smoke test: `/health`, `/schedule` for tomorrow, latest ingest partition < 1 h old.
- CI: deploy on tag (stretch: Workload Identity Federation, no JSON keys).
**Accept:** live URL returns tomorrow's schedule; jobs have run on schedule ≥ 2 days; monthly cost estimate in README (target < £5).
**Check yourself:** Which permissions does each job actually need? What happens if the plan job runs before that morning's ingest finishes?

### Stage 6 — Event-driven telemetry, optional (1 day)
**Learn:** Pub/Sub; at-least-once delivery; idempotent consumers; late/out-of-order data. Closest to how a real flexibility platform ingests device data.
**Build:**
- `flexcast simulate-fleet`: publishes battery telemetry (asset_id, ts, SoC, power) to a topic, with injected duplicates and out-of-order messages.
- `flexcast consume`: pull subscriber; dedupes on (asset_id, ts); writes micro-batches to storage.
- Plan job uses each asset's latest SoC as the LP's initial condition.
- Local first: Pub/Sub emulator in compose (official Google Cloud CLI emulators image, `PUBSUB_EMULATOR_HOST`), integration test for dedupe and ordering. Then deploy topic + consumer.
**Accept:** integration test proves duplicates don't double-count and late messages don't overwrite newer state.
**Check yourself:** Why is "exactly once" the wrong goal? Where does idempotency live in your design?

### Stage 7 — Write-up (½ day)
README, in this order:
1. One-paragraph problem + headline result (£ and % captured vs naive)
2. Architecture diagram (local and GCP)
3. Results table + chart
4. Testing approach (the four layers) and why local-first
5. Design decisions and what was deliberately not built
6. Limitations and next steps (portfolio bidding, BM participation, asset availability forecasting, price impact)
7. Run locally in 3 commands: `make up`, `make backtest`, open `/docs`

---

## 6. Non-goals
- Front end beyond FastAPI docs
- Elexon IRIS streaming — batch every 30 min is sufficient for a D-1 decision
- Kubernetes, Airflow, Kafka, MLflow, feature stores, BigQuery
- Multi-asset portfolio optimisation (next step only)

## 7. Definition of done
- Clean checkout with only Docker installed → `make up && make test && make itest && make backtest` pass
- Deployed, scheduled, producing daily schedules on GCP
- README tells the £ story in the first screen
