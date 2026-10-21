# Experiments

Generated from `log.jsonl` by `flexcast.experiments`; do not edit by hand.
£ metrics are per 10 kWh battery per year on the stated window.

| name | description | gbp_per_year | pct_of_perfect | mae | pinball | coverage_80 | window | git | when |
|---|---|---|---|---|---|---|---|---|---|
| baseline | Stage 2 model: 23 features, default LightGBM settings, no calibration | 32.55 | 12.02 | 30.59 | 9.79 | 0.66 | 2025-10-20..2026-10-19 | 0450374 | 2026-10-20 |
| solar | adds NESO embedded solar + wind forecasts (4 inputs), adopted on validation | 33.93 | 12.53 | 30.34 | 9.67 | 0.67 | 2025-10-20..2026-10-19 | bc96167 | 2026-10-20 |
| tuned | tuned LightGBM settings (notebook 04): strong regularisation, 150 rounds | 36.61 | 13.52 | 30.46 | 9.60 | 0.69 | 2025-10-20..2026-10-19 | cdf4eb2 | 2026-10-21 |
| calibrated | production strategy: tuned model + rolling calibration (90d, band only) + risk-aware alpha chosen on validation | 36.61 | 13.52 | 30.46 | 9.28 | 0.77 | 2025-10-20..2026-10-19 | 22903eb | 2026-10-21 |
