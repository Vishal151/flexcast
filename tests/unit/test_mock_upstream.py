import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mock_upstream.app import create_app

FIXTURES = Path(__file__).parents[1] / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(FIXTURES))


@pytest.mark.parametrize("entry", MANIFEST["entries"], ids=lambda e: e["label"])
def test_serves_every_manifest_entry_byte_for_byte(
    client: TestClient, entry: dict[str, Any]
) -> None:
    r = client.get(f"/{entry['source']}{entry['path']}", params=entry["params"])
    assert r.status_code == 200
    assert r.content == (FIXTURES / entry["file"]).read_bytes()
    assert r.headers["content-type"] == entry["content_type"]


def test_unknown_params_are_404_not_a_near_match(client: TestClient) -> None:
    r = client.get("/elexon/balancing/settlement/system-prices/2030-01-01")
    assert r.status_code == 404
    e = MANIFEST["entries"][0]
    r = client.get(f"/{e['source']}{e['path']}", params={**e["params"], "extra": "1"})
    assert r.status_code == 404


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "fixtures": len(MANIFEST["entries"])}
