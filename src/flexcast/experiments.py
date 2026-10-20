"""A lightweight experiment log kept in the repo: one JSON line per experiment plus a generated
comparison table (experiments/README.md). Enough to answer "which version was better, and why"
without running an experiment-tracking server."""

import json
from datetime import UTC, datetime
from typing import Any

from flexcast.storage import Storage

LOG = "log.jsonl"
TABLE = "README.md"
METRICS = ("gbp_per_year", "pct_of_perfect", "mae", "pinball", "coverage_80")


def read_log(store: Storage) -> list[dict[str, Any]]:
    if not store.exists(LOG):
        return []
    return [json.loads(line) for line in store.read_text(LOG).splitlines() if line.strip()]


def render_table(entries: list[dict[str, Any]]) -> str:
    header = ["name", "description", *METRICS, "window", "git", "when"]
    lines = [
        "# Experiments",
        "",
        "Generated from `log.jsonl` by `flexcast.experiments`; do not edit by hand.",
        "£ metrics are per 10 kWh battery per year on the stated window.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for e in entries:
        m = e["metrics"]
        cells = [e["name"], e["description"]]
        cells += [f"{m[k]:.2f}" if isinstance(m.get(k), int | float) else "" for k in METRICS]
        cells += [e["window"], e["git_sha"], e["when"][:10]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def log_experiment(
    store: Storage,
    name: str,
    description: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    window: str,
    git_sha: str,
    when: str | None = None,
) -> None:
    entry = {
        "name": name,
        "description": description,
        "config": config,
        "metrics": metrics,
        "window": window,
        "git_sha": git_sha,
        "when": when or datetime.now(UTC).isoformat(timespec="seconds"),
    }
    entries = [*read_log(store), entry]
    store.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries), LOG)
    store.write_text(render_table(entries), TABLE)
