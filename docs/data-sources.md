# Data sources

Verified 2026-10-20 against the live Elexon OpenAPI spec
(`https://data.elexon.co.uk/swagger/v1/swagger.json`, 322 paths) and live responses.
Fixtures for every request below are in `tests/fixtures/` (see `manifest.json`); regenerate with
`uv run python scripts/record_fixtures.py`, inspect with `uv run python scripts/explore_api.py [label]`.

Sample days: normal `2025-06-11` (48 periods), spring DST `2025-03-30` (46), autumn DST `2025-10-26` (50).

## Common conventions

- All Elexon timestamps are UTC with a `Z` suffix. `settlementDate` is a **London** date: on
  2025-06-11, period 1 starts `2025-06-10T23:00Z`.
- Elexon `/datasets/*` responses are `{"data": [...]}`; Octopus is `{"count", "next", "previous", "results"}`.
- Elexon prices are **£/MWh**; Octopus Agile is **p/kWh** (exc/inc VAT). 1 p/kWh = £10/MWh.
- No API keys needed for either source.

## Elexon — base `https://data.elexon.co.uk/bmrs/api/v1`

### System prices (forecast target) — verified
`GET /balancing/settlement/system-prices/{settlementDate}`
- One row per settlement period; 48/46/50 confirmed.
- Fields used: `settlementDate`, `settlementPeriod`, `startTime`, `systemSellPrice`, `systemBuyPrice`,
  `createdDateTime`, `priceDerivationCode`, `netImbalanceVolume`. 22 fields in total (volumes, adjustments, flags).
- SSP == SBP on all sample days (single imbalance price). We store both anyway.
- `createdDateTime` is about 24 h 45 min after each period starts. **Open question:** the endpoint returns one
  row per period (the latest settlement run?). Revisions are not exposed here. Check whether a run-type
  field or another endpoint (`DISEBSP` dataset) exposes runs before Stage 2.

### MID — Market Index Data (feature, lagged) — verified
`GET /datasets/MID?from=<utc>&to=<utc>[&dataProviders=APXMIDP]`
- `from`/`to` are **instants** (inclusive). A bare date `from=2025-10-26&to=2025-10-26` returns a single
  timestamp (00:00Z, which is SP3 on that autumn DST day). Query with the London-day UTC bounds instead.
- Fields: `startTime`, `dataProvider`, `settlementDate`, `settlementPeriod`, `price`, `volume`.
- Two providers: `APXMIDP` and `N2EXMIDP`. **N2EX had only 46 of 50 periods on the autumn DST day.**
  Use `APXMIDP`.

### NDF — day-ahead national demand forecast (feature, vintage) — verified
`GET /datasets/NDF?publishDateTimeFrom=<utc>&publishDateTimeTo=<utc>`
- Filters by **publish time only**. This makes point-in-time history possible.
- **Window cap: max 1 day per request** (HTTP 400 otherwise). The client must chunk.
- Fields: `publishTime`, `startTime`, `settlementDate`, `settlementPeriod`, `demand` (MW), `boundary` (`N` only).
- Cadence: a new vintage roughly every 30 min (`:16`/`:46`, sometimes off by a minute or more: `10:23`, `10:48`).
  About 48 vintages per London day.
- Horizon: each vintage covers the rest of today plus part or all of the next 1–2 days. How far it reaches
  depends on the time of day (see "Vintage visibility" below).
- Natural key: (`publishTime`, `startTime`). No duplicates within a vintage on the sample days.
- Alternatives in the spec: `/forecast/demand/day-ahead/history?publishTime=` (one vintage) and
  `/forecast/demand/day-ahead/evolution?settlementDate=&settlementPeriod=` (all vintages for one period).
  Not needed if we ingest the dataset by publish window.

### WINDFOR — wind generation forecast (feature, vintage) — verified, **differs from SPEC**
`GET /datasets/WINDFOR?publishDateTimeFrom=<utc>&publishDateTimeTo=<utc>`
- **Hourly** `startTime`, not half-hourly. Features must upsample to 30 min (forward-fill or interpolate:
  a modelling choice for Stage 2).
- Fields: `publishTime`, `startTime`, `generation` (MW). No settlement date/period fields.
- Cadence: **8 vintages/day**. Observed UTC times: 23:30, 03:30, 05:30, 08:30, 10:30, 12:30, 16:30, 19:30
  on BST days; 02:30, 04:30, 07:30, 09:30, 11:30, 15:30, 18:30, 22:30 on 2025-03-28/29 (GMT). The schedule
  is neither fixed in UTC nor in London time. Don't hardcode it; always filter on `publishTime`.
- Horizon: from about 1 day before publish to about D+2 20:00Z.

### FUELHH — generation by fuel (feature, lagged / EDA) — verified
`GET /datasets/FUELHH?settlementDateFrom=<date>&settlementDateTo=<date>`
- Also supports publish-time filtering and `fuelType` filters.
- 20 fuel types per period (incl. interconnectors `INT*`). `publishTime` = `startTime` + 30 min.

### TEMP — temperature (optional feature) — verified, low value
`GET /datasets/TEMP?publishDateTimeFrom=<utc>&publishDateTimeTo=<utc>`
- **One row per day** (`measurementDate`, `temperature`), published around 15:45Z. Daily resolution only.

## Octopus — base `https://api.octopus.energy/v1`

### Agile unit rates (Stage 1 optimiser validation only) — verified
- Discover the product: `GET /products/?available_at=<utc>` and filter codes containing `AGILE` with
  `direction == IMPORT`. Current import product: `AGILE-24-10-01`. Others in the listing: `AGILE-BB-*` (business),
  `AGILE-OUTGOING-*` (export).
- Tariff code per region: `GET /products/{code}/` → `single_register_electricity_tariffs._C.direct_debit_monthly.code`
  = `E-1R-AGILE-24-10-01-C` (region C = London).
- Rates: `GET /products/{code}/electricity-tariffs/{tariff}/standard-unit-rates/?period_from=<utc>&period_to=<utc>`
- Fields: `valid_from`, `valid_to`, `value_exc_vat`, `value_inc_vat` (p/kWh). Returned **newest first**.
- Negative rates do occur (min −5.08 p/kWh on 2025-03-30).
- Publishes around 16:00 London for the next day (per SPEC; not re-verified, since all sample days are historic).
- `period_from` must be the London-midnight instant. On the spring DST day that is `00:00Z`, not `23:00Z` the
  day before.

## Vintage visibility at decision time

_To be filled in after the Stage 0 "check yourself" discussion._

## Proposed SPEC §2 edits
1. WINDFOR is hourly; note upsampling to half-hourly.
2. MID: use `APXMIDP` (N2EX has gaps).
3. NDF/WINDFOR publish-window queries are capped at 1 day.
4. TEMP is daily; keep it optional, low priority.

## Attribution
Contains BMRS data © Elexon Limited copyright and database right.
