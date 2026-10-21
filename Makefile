lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest

# make ingest START=2025-06-01 END=2025-06-07  (no args: yesterday and today)
ingest:
	uv run flexcast ingest $(if $(START),--start $(START)) $(if $(END),--end $(END))

# Reproducible results table: fixed windows over the Stage 1 backfill.
backtest:
	FLEXCAST_GIT_SHA=$$(git rev-parse --short HEAD) uv run flexcast backtest \
		--train-start 2024-10-20 --test-start 2025-10-20 --test-end 2026-10-19

docker:
	docker build -t flexcast:local .

# Build the image, start gcs + upstream-mock + api, and wait until all are healthy.
# UP_FLAGS=--no-build lets CI reuse the image it already built with a layer cache.
UP_FLAGS ?= --build
up:
	docker compose up -d --wait $(UP_FLAGS)

down:
	docker compose down -v --remove-orphans

itest:
	uv run pytest -m integration tests/integration

# Opt-in: hits the real Elexon and Octopus APIs (weekly in CI to catch upstream drift).
live:
	uv run pytest -m live tests/live

# Load the fixture days into the running stack, train and plan, so the API has something to show.
DEMO_RUN = docker compose run --rm -q
demo:
	for d in 2025-03-30 2025-06-11 2025-10-26; do \
		$(DEMO_RUN) ingest --start $$d --end $$d --datasets system_prices,mid,fuelhh,agile; done
	$(DEMO_RUN) ingest --start 2025-06-09 --end 2025-06-10 --datasets ndf,windfor,embedded
	$(DEMO_RUN) train --as-of 2025-10-28T12:00:00+00:00
	$(DEMO_RUN) plan --date 2025-06-11
	@echo "API: http://localhost:8080/docs  e.g. /schedule?date=2025-06-11"

# Re-run the model-development notebooks (needs the ./data backfill and a backtest run).
# The .py files are the source; jupytext syncs them into .ipynb, which keep the outputs.
notebooks:
	uv run jupytext --sync notebooks/*.py
	uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/*.ipynb

.PHONY: lint test ingest backtest docker up down itest live demo notebooks
