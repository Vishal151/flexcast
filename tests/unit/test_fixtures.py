"""Characterisation tests for recorded upstream fixtures.

These pin down facts about the real data that later stages rely on: period counts on DST days,
forecast vintages carrying a publish time, and every manifest entry being replayable.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parents[1] / "fixtures"
MANIFEST = FIXTURES / "manifest.json"

DAYS = {
    "normal": ("2025-06-11", 48),
    "spring_dst": ("2025-03-30", 46),
    "autumn_dst": ("2025-10-26", 50),
}


def manifest() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = json.loads(MANIFEST.read_text())["entries"]
    return entries


def load(label: str) -> Any:
    (entry,) = [e for e in manifest() if e["label"] == label]
    return json.loads((FIXTURES / entry["file"]).read_text())


def rows(label: str) -> list[dict[str, Any]]:
    body = load(label)
    data: list[dict[str, Any]] = body["data"] if "data" in body else body["results"]
    return data


def test_every_manifest_entry_has_a_parseable_file() -> None:
    entries = manifest()
    assert entries
    for e in entries:
        assert {"label", "source", "method", "path", "params", "file"} <= e.keys()
        json.loads((FIXTURES / e["file"]).read_text())


def test_labels_are_unique() -> None:
    labels = [e["label"] for e in manifest()]
    assert len(labels) == len(set(labels))


@pytest.mark.parametrize("kind", DAYS)
def test_system_prices_period_count(kind: str) -> None:
    day, expected = DAYS[kind]
    r = rows(f"system_prices_{kind}")
    assert {x["settlementDate"] for x in r} == {day}
    assert sorted(x["settlementPeriod"] for x in r) == list(range(1, expected + 1))


@pytest.mark.parametrize("kind", DAYS)
def test_agile_period_count(kind: str) -> None:
    _, expected = DAYS[kind]
    assert len(rows(f"agile_{kind}")) == expected


@pytest.mark.parametrize("dataset", ["ndf", "windfor"])
def test_forecasts_have_multiple_utc_vintages(dataset: str) -> None:
    labels = [e["label"] for e in manifest() if e["label"].startswith(f"{dataset}_")]
    assert labels
    for label in labels:
        r = rows(label)
        publish = {x["publishTime"] for x in r}
        assert len(publish) > 1, label
        for p in publish:
            assert datetime.fromisoformat(p).utcoffset() is not None, p


def test_ndf_target_day_has_one_row_per_period_per_vintage() -> None:
    """No vintage should contain the same target period twice (natural key sanity)."""
    for e in manifest():
        if e["label"].startswith("ndf_"):
            keys = Counter((x["publishTime"], x["startTime"]) for x in rows(e["label"]))
            assert max(keys.values()) == 1, e["label"]
