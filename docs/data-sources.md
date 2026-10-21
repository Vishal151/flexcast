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
  `direction == IMPORT`. Current import product: `AGILE-24-10-01`. Also filter `brand == OCTOPUS_ENERGY`: `AGILE-BB-*` is the
  same tariff under the Bulb brand (not business). `AGILE-OUTGOING-*` is export.
- Tariff code per region: `GET /products/{code}/` → `single_register_electricity_tariffs._C.direct_debit_monthly.code`
  = `E-1R-AGILE-24-10-01-C` (region C = London).
- Rates: `GET /products/{code}/electricity-tariffs/{tariff}/standard-unit-rates/?period_from=<utc>&period_to=<utc>`
- Fields: `valid_from`, `valid_to`, `value_exc_vat`, `value_inc_vat` (p/kWh). Returned **newest first**.
- Negative rates do occur (min −5.08 p/kWh on 2025-03-30).
- Publishes around 16:00 London for the next day (per SPEC; not re-verified, since all sample days are historic).
- `period_from` must be the London-midnight instant. On the spring DST day that is `00:00Z`, not `23:00Z` the
  day before.

## Vintage visibility at decision time

Worked example from the fixtures: target day **2025-06-11**, decision time **D-1 09:00 London =
2025-06-10T08:00Z** (BST; in GMT it would be 09:00Z).

### NDF
74 vintages published before the target day touch 2025-06-11, from 2025-06-09T07:45Z to
2025-06-10T22:47Z. About 47 more are published during the target day itself; those are irrelevant
for a D-1 decision and were not recorded. The vintages come in three kinds:

| Published (UTC) | Covers on D | Role |
|---|---|---|
| every ~30 min (`:16`/`:46`) until ~10:16 on D-1 | SP1–10 | rolling: rest of today + first ~5 h of tomorrow |
| **07:45 once daily** | SP11–48 of D+1 (and SP1–10 of D+2) | the once-a-day run that fills in the rest of tomorrow |
| every ~30 min from ~10:47 on D-1 | SP1–48 | full day-ahead coverage |

At 2025-06-10T08:00Z:
- the latest vintage overall is **07:45Z**, but it covers only SP11–48
- the latest vintage covering SP1–10 is **07:16Z**
- **no single vintage covers all 48 periods.** The first complete one is 10:47Z, after the decision.

### WINDFOR
The latest vintage before 08:00Z is **05:30Z**, and it covers all 24 hours of the target day. No
stitching needed.

### Rules for Stage 2 (`features.py`)
1. **Select per target period, not per vintage:** for each `startTime`, take the row with the max
   `publishTime < decision_time`. Picking "the latest vintage" and using only its rows would silently
   drop SP1–10 or SP11–48.
2. **Margins vary by season.** In BST the 07:45Z run lands 15 min before the deadline; in GMT it lands
   75 min before. If the run is late, SP11–48 falls back to the previous day's 07:45Z run (24 h
   older). Record the age of the forecast used (decision_time − publishTime) as a feature or
   diagnostic.
3. **Leakage test:** every NDF/WINDFOR row used must have `publishTime < decision_time`. The 10:47Z
   vintage is published on D-1 but after the decision, so using it is leakage.

### Alternative (not adopted)
A decision time after ~11:00 London would give one complete NDF vintage and remove the stitching.
We keep 09:00 (SPEC default, closer to real day-ahead market deadlines) and handle the stitching
explicitly. It stays configurable.

## Candidate inputs for the model stage (verified 2026-10-20)

The test for each: at the decision time (D-1 09:00 London; for target 2025-06-11 that is
2025-06-10T08:00Z), is there a vintage covering the whole target day?

| Source | What | Full target day at decision? | Verdict |
|---|---|---|---|
| Elexon `DGWS` | day-ahead wind + solar | **No**: published ~16:45 D-1 | unusable at 09:00 |
| Elexon `IMBALNGC` (boundary `N`) | indicated imbalance | **No**: SP1-10 only (rolling run) | partial; full-day run ~10:47 |
| Elexon `MELNGC` (boundary `N`) | indicated margin | **No**: SP1-10 only | partial |
| Elexon `LOLPDRM` | loss-of-load probability, de-rated margin | **No**: ~10 periods ahead | partial |
| NESO `embedded-wind-and-solar-forecasts` | embedded solar + wind, MW | **Yes**: hourly vintages (23/day), latest 07:12 covered SP1-48 | **use** |
| NESO `day-ahead-power-exchange-prices-nordpool` | N2EX GB day-ahead auction price, £/MWh, hourly | target/framing, see below | **use** |

Takeaway: at 09:00 D-1 almost nothing official covers all of tomorrow (NDF and WINDFOR are the
exceptions via their early runs). The decision time, more than the model, limits what can be known.

### NESO embedded wind and solar forecast
- CKAN datastore API: `https://api.neso.energy/api/3/action/datastore_search_sql`; NESO Open Data Licence.
- Yearly archive resources (2019-2026) keep every vintage via `Forecast_Datetime` (~5.4M rows/year);
  the live resource holds only the latest snapshot (no vintage column), so live ingest must record
  when it observed each value.
- Fields: `SETTLEMENT_DATE`, `SETTLEMENT_PERIOD`, `EMBEDDED_SOLAR_FORECAST`, `EMBEDDED_SOLAR_CAPACITY`,
  `EMBEDDED_WIND_FORECAST`, `EMBEDDED_WIND_CAPACITY` (MW), `Forecast_Datetime`.
- `Forecast_Datetime` has no timezone. **Treat as UTC until confirmed**: if it is really London time,
  reading it as UTC makes a vintage look later than it was (safe); the reverse would leak.
- `DATE_GMT` + `TIME_GMT` = the **end** of the settlement period in UTC (checked in summer and winter).
  The client derives `start_time` from settlement date/period with our helper and rejects rows where
  start + 30 min disagrees with `TIME_GMT`.
- **Availability = `Forecast_Datetime` + 2 h.** The current-year archive is kept up to date, but a vintage
  appears some 25-85 min after its timestamp (on 2026-10-20 at 21:17Z the latest was 19:52); 2 h covers
  that and the unconfirmed timezone. At a 09:00 BST decision this uses the ~05:12Z vintage.
- Each vintage forecasts ~13 days ahead (~620 rows); ingest keeps up to 2 days ahead (MAX_HORIZON_DAYS).
- Backfill quirks (2024-10-18..2026-10-19, 17 MB): **no vintages published 2025-05-09..11** (archive gap;
  targets 2025-05-11/12 get no solar inputs, which LightGBM treats as missing). On **2026-04-20/21** a few
  vintages stamped exactly 00:00 have `TIME_GMT` in **London time** (an hour off), likely NESO's forecast
  system migration; the start-time cross-check rejected those ~190 rows (kept in `rejects/embedded/`).
- Example: vintage 2025-06-10T07:12 forecast 11,949 MW solar for 2025-06-11 13:00 (capacity 19,458 MW).

### NESO N2EX GB day-ahead prices
- Hourly (`Date`, `Delivery Period` "HH:MM - HH:MM", `Price` £/MWh), 2021-09-20 onward; tomorrow's
  prices were already present on 2026-10-20 evening. Same API and licence as above.
- **No publish timestamp.** Auction timing per Nord Pool: gate closure **09:50 UK time on D-1**, results
  **by 10:00**, usually within minutes; a second auction can delay results if price thresholds are hit.
  NESO's republication time: being measured (see journal).
- **`Date` + `Delivery Period` are UTC, not London time** (verified 2026-10-21). Every day has exactly
  24 rows, including both clock-change days (London days have 23/25 hours), and on summer days the
  series lines up with Octopus Agile (which is derived from day-ahead prices, with exact UTC times) far
  better read as UTC (residual sd 1.96 p/kWh, correlation 0.967) than as London time (2.74, 0.935) or
  CET (correlation 0.857 over all months). Read as London time, every summer price would be an hour off.
- No duplicate (Date, Delivery Period) rows in 2021-09-20..2026-10-21. The current day can be partial
  (2026-10-21 at 09:35Z had 22 rows, to 21:00-22:00).

## SPEC §2 edits (applied)
WINDFOR hourly, MID provider `APXMIDP`, 1-day publish-window cap, MID instants, TEMP daily/low priority.

## Attribution
Contains BMRS data © Elexon Limited copyright and database right.
