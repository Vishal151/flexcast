# Journal

One entry per stage: what was built, what surprised us, open questions.

## Stage 0 — Explore the data
- Built `scripts/explore_api.py` and `scripts/record_fixtures.py` over one shared request list, plus a fixture manifest for respx and mock_upstream. Recorded 26 responses (3.9 MB) for a normal day and both 2025 DST days. Every SPEC §2 Elexon path exists in the live OpenAPI spec.
- Surprises: WINDFOR is hourly with 8 vintages/day on a schedule that is neither UTC- nor London-fixed. NDF publish-window queries are capped at 1 day. MID `from`/`to` are instants, not dates. N2EX MID drops periods on the autumn DST day. My own hand-written Agile window was an hour off on the spring DST day, which is the case for deriving windows from London dates in code.
- Open: does the system-prices endpoint expose settlement-run revisions (`createdDateTime` is ~D+1)? How much of the target day does the latest NDF vintage before D-1 09:00 actually cover? That's the Check yourself question, and it may push decision time or feature design.
