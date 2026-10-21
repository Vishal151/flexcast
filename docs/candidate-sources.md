# Candidate data sources (not yet used)

Inputs a commercial GB price forecaster would add, checked on 2026-10-21 for free, automatable access.
"Probed" means we fetched real data from the endpoint; "unverified" means documentation only.

| Driver | Source | Free / automatable | Point-in-time? | Status |
|---|---|---|---|---|
| Gas price (sets the price most hours) | ONS republication of National Gas **System Average Price**, daily since 2018 ([page](https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas)) | Yes; weekly .xlsx, link scraped from the page | Lagged ~6 days (file of 15 Oct ended 9 Oct); slow-moving, so still informative for the price *level* | Probed |
| Gas price, same day | National Gas [Gas Data Portal](https://nationalgas.com/our-businesses/operational-data/gas-data-portal) CSV download (`PUBOB603`), no key | Yes | Gas day d at ~12:40 UK on d+1 | **Ingested**, tested in the model |
| Carbon price | ICE UK Allowance auctions, fortnightly ([ICE](https://www.ice.com/emissions/auctions/uk-emission-allowances)); gov.uk [annual summaries](https://www.gov.uk/government/collections/functioning-of-the-uk-carbon-market) | Auction results free but no confirmed download; daily futures prices are paid | Fortnightly | Unverified |
| Plant outages | Elexon REMIT messages (`/remit/list/by-publish`) and unit availability 2–14 days ahead (`/datasets/UOU2T14D`, has `publishTime`) | Yes, same API as today, no key | Yes, publish times given | **FOU2T14D (by fuel) ingested**, tested in the model; REMIT not used |
| Interconnector flows | Elexon `FUELHH` (`INTFR`, `INTNOR`, ... fuel types) | Yes | Actuals only (lagged like prices) | Probed |
| Neighbouring countries (French nuclear, Norwegian hydro, continental prices) | ENTSO-E Transparency Platform ([entsoe-py](https://pypi.org/project/entsoe-py/)) | Free; needs registration + emailed token request | Yes for day-ahead prices and outage documents | Unverified; GB itself no longer reports there since Brexit |
| Weather (wind at turbine height, temperature, many forecast runs) | Open-Meteo [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) (fixed 1–7 day lead) and [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) | Free for **non-commercial** use only | Previous Runs: true day-ahead values from ~2024 (2023 came back empty); [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) goes back to 2021 but stitches each run's first hours, so it is slightly better than a real day-ahead forecast (mild leakage) | Probed |
| Grid bottlenecks | NESO `day-ahead-constraint-flows-and-limits`, `thermal-constraint-costs` | Yes (same API as embedded forecasts) | History of past vintages not yet checked | Probed (dataset exists) |
| Demand | NESO `1-day-ahead-demand-forecast` (we already use Elexon NDF) | Yes | Vintage history not yet checked | Probed (dataset exists) |
| Holidays | gov.uk [bank-holidays.json](https://www.gov.uk/bank-holidays.json) | Yes | Known years ahead | Probed |

## Why gas is the obvious next step
Gas plants are usually the most expensive generator still needed, so they set the GB price most hours. Gas
moves the *level* of a whole day; the battery mostly needs the *shape*, which yesterday's auction prices
already carry. Gas would matter most in regime changes (e.g. 2022), where a weekly-retrained model lags.

## Location
GB has a single national wholesale price; the July 2025 market review rejected zonal pricing in favour of
"reformed national pricing" ([summary](https://cms.law/en/gbr/legal-updates/rema-summer-update-zonal-pricing-out-reformed-national-pricing-in)).
So the price forecast is rightly national. A battery's *location* still changes what it earns, through:
the home's own solar and usage, regional network charges in retail tariffs (Agile differs across 14
regions), and local flexibility payments from distribution network operators in constrained areas.
