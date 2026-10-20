"""Integration tests: real containers, real GCS client, emulated GCS, mocked upstream.

Needs `make up`. Jobs run exactly as an operator would run them (`docker compose run --rm <job>`),
pointed at a dedicated bucket that is recreated for each session. The host only *reads* the
bucket (through the same Storage/gcsfs code) to check what the containers wrote.
"""

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EMULATOR = "http://localhost:4443"
BUCKET = "flexcast-itest"

# Host-side gcsfs talks to the emulator the same way the containers do (see compose.yaml).
os.environ.update(
    STORAGE_EMULATOR_HOST=EMULATOR,
    FSSPEC_GCS_TOKEN="anon",
    GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT="false",
)
import fsspec  # noqa: E402

fsspec.config.set_conf_env(fsspec.config.conf)  # pick up FSSPEC_GCS_TOKEN if fsspec was loaded

from flexcast.storage import Storage  # noqa: E402


def compose(*args: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )


def run_job(job: str, *args: str) -> subprocess.CompletedProcess[str]:
    """`docker compose run --rm <job> <args>` against the test bucket."""
    return compose("run", "--rm", "-q", "-e", f"FLEXCAST_DATA_URI=gs://{BUCKET}", job, *args)


def ok(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, f"exit {result.returncode}\n{result.stdout}\n{result.stderr}"
    return result.stdout + result.stderr


def _gcs(method: str, path: str, body: dict[str, str] | None = None) -> None:
    req = urllib.request.Request(
        f"{EMULATOR}/storage/v1{path}",
        method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"content-type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise


def _drop_bucket() -> None:
    try:
        with urllib.request.urlopen(f"{EMULATOR}/storage/v1/b/{BUCKET}/o", timeout=10) as r:
            names = [o["name"] for o in json.load(r).get("items", [])]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return
        raise
    for name in names:
        _gcs("DELETE", f"/b/{BUCKET}/o/{urllib.parse.quote(name, safe='')}")
    _gcs("DELETE", f"/b/{BUCKET}")


@pytest.fixture(scope="session")
def stack() -> None:
    running = compose("ps", "--status", "running", "--services").stdout.split()
    missing = {"gcs", "upstream-mock"} - set(running)
    if missing:
        pytest.fail(f"compose services not running: {sorted(missing)}; run `make up` first")


@pytest.fixture(scope="session")
def bucket(stack: None) -> Iterator[Storage]:
    _drop_bucket()
    _gcs("POST", "/b", {"name": BUCKET})
    yield Storage(f"gs://{BUCKET}")
    _drop_bucket()


def hashes(storage: Storage, prefix: str) -> dict[str, str]:
    return {p: hashlib.sha256(storage.read_bytes(p)).hexdigest() for p in storage.list(prefix)}


def wait_for(url: str, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except (urllib.error.URLError, ConnectionError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)
