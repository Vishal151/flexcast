lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest

.PHONY: lint test
