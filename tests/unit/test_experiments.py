import json
from pathlib import Path

from flexcast.experiments import log_experiment, read_log, render_table
from flexcast.storage import Storage


def test_log_appends_and_renders_table(tmp_path: Path) -> None:
    store = Storage(f"file://{tmp_path}")
    log_experiment(
        store,
        "baseline",
        "current model",
        {"num_leaves": 31},
        {"gbp_per_year": 32.55, "mae": 30.6},
        window="2025-10-20..2026-10-19",
        git_sha="abc",
        when="2026-10-20T20:00:00+00:00",
    )
    log_experiment(
        store,
        "solar",
        "adds solar",
        {"num_leaves": 31},
        {"gbp_per_year": 40.0, "mae": 28.0},
        window="2025-10-20..2026-10-19",
        git_sha="def",
        when="2026-10-21T20:00:00+00:00",
    )
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert [json.loads(line)["name"] for line in lines] == ["baseline", "solar"]
    assert [e["name"] for e in read_log(store)] == ["baseline", "solar"]
    table = (tmp_path / "README.md").read_text()
    assert table == render_table(read_log(store))
    assert "| solar | adds solar |" in table and "40.00" in table and "def" in table


def test_read_log_when_empty(tmp_path: Path) -> None:
    assert read_log(Storage(f"file://{tmp_path}")) == []
