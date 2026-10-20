lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest

# make ingest START=2025-06-01 END=2025-06-07  (no args: yesterday and today)
ingest:
	uv run flexcast ingest $(if $(START),--start $(START)) $(if $(END),--end $(END))

.PHONY: lint test ingest
