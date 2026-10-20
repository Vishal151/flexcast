"""Replay recorded upstream fixtures through respx. The manifest is the single source of truth,
shared with mock_upstream (Stage 3). Unmatched requests fail loudly: no silent network access."""

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def upstream() -> Iterator[respx.MockRouter]:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        for e in manifest["entries"]:
            url = manifest["base_urls"][e["source"]] + e["path"]
            body = (FIXTURES / e["file"]).read_bytes()
            router.route(method=e["method"], url=url, params__eq=e["params"]).mock(
                return_value=httpx.Response(
                    200, content=body, headers={"content-type": "application/json"}
                )
            )
        yield router
