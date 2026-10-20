# Journal

One entry per stage: what was built, what surprised us, open questions.

## Stage 0 — Explore the data
- Built `scripts/explore_api.py` and `scripts/record_fixtures.py` over one shared request list, plus a fixture manifest for respx and mock_upstream. Recorded 26 responses (stored as indented JSON for reviewable diffs, 5.4 MB) for a normal day and both 2025 DST days. Every SPEC §2 Elexon path exists in the live OpenAPI spec.
- Surprises: WINDFOR is hourly with 8 vintages/day on a schedule that is neither UTC- nor London-fixed. NDF publish-window queries are capped at 1 day. MID `from`/`to` are instants, not dates. N2EX MID drops periods on the autumn DST day. My own hand-written Agile window was an hour off on the spring DST day, which is the case for deriving windows from London dates in code.
- Open: does the system-prices endpoint expose settlement-run revisions (`createdDateTime` is ~D+1)? How much of the target day does the latest NDF vintage before D-1 09:00 actually cover? That's the Check yourself question, and it may push decision time or feature design.

## Stage 1 — Core library, local only
- Built config (env-only), fsspec storage (atomic writes), a settlement helper checked against Elexon's own mapping for 144 real periods, Elexon/Octopus clients (retry/backoff, chunking to the 1-day/7-day caps, pydantic validation that also checks settlement date/period against start_time), idempotent ingest (merge on natural key, deterministic Parquet, rejects kept in `rejects/`) and the battery MILP (PuLP + HiGHS). Backfilled 730 days × 6 datasets (40 MB) in ~3.5 min with no failures; a live re-ingest is byte-identical.
- Surprises: PuLP 4 no longer bundles CBC and changed its API (`prob.add_variable`, `LpSolveStatus`), so we added `highspy`. `AGILE-BB-*` is the Bulb-branded Agile, not business. With perfect foresight on system prices a 10 kWh battery makes £278/yr. Degradation is the biggest lever (£656/yr at 0p/kWh vs £278 at 2p). 90%→80% efficiency costs about 20% (£278→£221). Agile gives £648/yr: its retail formula amplifies wholesale spreads and adds a 16:00–19:00 uplift, so it's not a market price.
- MILP vs LP: the binary removes simultaneous charge/discharge (39 of 48 periods in the relaxed LP at −£1000/MWh) but only changes £ by 2p there. The battery can still "earn from losses" by alternating periods, which is physically allowed.
- Open: the 2p/kWh degradation default needs a defensible source before Stage 2 results; Agile export would be paid at the Outgoing rate, so the symmetric-price Agile number overstates value.
