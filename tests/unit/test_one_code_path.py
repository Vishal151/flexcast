"""CLAUDE.md: local, compose and GCP differ only in env vars. Guard against profile branches."""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).parents[2] / "src" / "flexcast"
FORBIDDEN = [
    r"STORAGE_EMULATOR_HOST",  # emulator wiring belongs in compose.yaml, not code
    r"FSSPEC_GCS_TOKEN",
    r"upstream-mock",
    r"fake-gcs",
    r"\bprofile\b",
    r"localhost",
]


@pytest.mark.parametrize("pattern", FORBIDDEN)
def test_source_has_no_environment_specific_code(pattern: str) -> None:
    hits = [
        f"{p.relative_to(SRC)}:{i}"
        for p in SRC.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(pattern, line)
    ]
    assert not hits, f"{pattern!r} found in {hits}"
