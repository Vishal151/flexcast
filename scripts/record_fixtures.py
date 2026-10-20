"""Record Stage 0 upstream responses to tests/fixtures/ and write manifest.json.

Bodies are saved byte-for-byte so respx (unit) and mock_upstream (integration) replay exactly
what the real API returned. The manifest maps (source, method, path, params) -> file.

    uv run python scripts/record_fixtures.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sources import BASE_URLS, requests

FIXTURES = Path(__file__).parents[1] / "tests" / "fixtures"


def main() -> None:
    entries = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for req in requests():
            resp = client.get(BASE_URLS[req.source] + req.path, params=req.params)
            resp.raise_for_status()
            file = f"{req.source}/{req.label}.json"
            out = FIXTURES / file
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(resp.content)
            entries.append(
                {
                    "label": req.label,
                    "source": req.source,
                    "method": "GET",
                    "path": req.path,
                    "params": req.params,
                    "file": file,
                    "content_type": resp.headers.get("content-type", "application/json"),
                }
            )
            print(f"{req.label:32s} {len(resp.content):>8d} bytes")
    manifest = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_urls": BASE_URLS,
        "entries": entries,
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
