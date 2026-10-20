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
up:
	docker compose up -d --build --wait

down:
	docker compose down -v --remove-orphans

itest:
	uv run pytest -m integration tests/integration

.PHONY: lint test ingest backtest docker up down itest
