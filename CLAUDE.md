# CLAUDE.md

Read `SPEC.md` first. Work **one stage at a time, in order**. Do not start the next stage until the current stage's acceptance criteria pass.

## Learning mode
This is a learning project. The goal is that I understand every layer, not just that it works.
- At the start of each stage, explain the key concepts for that stage's **Learn** line in ≤ 10 bullets, then propose the plan.
- For items marked **(I write)** below, don't implement them — give me the interface, the tests, and hints. Review my code when I'm done.
- After each stage, ask me the stage's **Check yourself** questions and correct my answers.
- Prefer explaining a trade-off over silently picking one.

**(I write)** items:
- Stage 1: the battery LP formulation in `optimise.py`
- Stage 2: the leakage test and `features.py::build`
- Stage 3: the first draft of the Dockerfile
- Stage 6: the consumer's dedupe logic

## Workflow per stage
1. Explain concepts, then plan: files to create/change, tests to write. Wait for approval.
2. Write tests before implementation where behaviour is specifiable (LP constraints, DST, leakage, idempotency, dedupe).
3. Implement the smallest thing that passes.
4. Run `make lint test` (and `make itest` from Stage 3). Fix, don't skip.
5. Append 3–5 lines to `docs/journal.md`: what was built, what surprised us, open questions.
6. Commit: `stage N: <summary>`.

## Stack
- Python 3.12, `uv`, `ruff` (lint + format), `mypy` (strict on `src/`), `pytest`, `respx`
- `httpx`, `pandas`, `pyarrow`, `duckdb`, `fsspec`, `gcsfs`, `lightgbm`, `cvxpy` (HiGHS) or `pulp`, `pydantic-settings`, `typer`, `fastapi`
- Stage 3+: Docker, docker compose, `fsouza/fake-gcs-server`
- Stage 6: `google-cloud-pubsub`, Pub/Sub emulator

## Conventions
- All timestamps UTC, tz-aware. Europe/London only for display and decision-time config.
- Settlement date/period derived from UTC timestamp via one tested helper. Never `period = hour*2 + 1`.
- **One code path across profiles.** Local, compose and GCP differ only in env vars (`FLEXCAST_DATA_URI`, upstream base URLs, `STORAGE_EMULATOR_HOST`). If you're tempted to write `if profile == ...`, stop and raise it.
- All I/O through `storage.py`. Never write to a hard-coded path or call the GCS client directly from business logic.
- Unit tests: no network, no Docker. Integration tests: marker `integration`, need `make up`. Live tests: marker `live`, opt-in.
- Fixtures are recorded once and shared by `respx` and `mock_upstream`.
- Functions taking `decision_time` must not read data published after it. If unsure, assert.
- Config via env only; no secrets in code. Nothing needs a secret except GCP auth (ADC locally, service account in GCP).
- Prefer boring code. No new dependency without a one-line justification in the plan.

## Commands
- `make lint` — ruff + mypy
- `make test` — unit tests
- `make up` / `make down` — compose stack
- `make itest` — integration tests against compose
- `make live` — live upstream smoke test
- `make ingest START=... END=...`, `make backtest`
- `make docker` — build image
- `make deploy ENV=dev`, `make smoke ENV=dev` (Stage 5)

## When unsure
- API shape: check the live OpenAPI spec at https://developer.data.elexon.co.uk/ and update `docs/data-sources.md`.
- Emulator behaviour differs from real GCP: note it in the journal and cover it in the Stage 5 smoke test.
- Modelling choice: pick the simpler option, note the alternative.
- Anything that changes the spec: propose the edit to `SPEC.md`; don't silently diverge.
