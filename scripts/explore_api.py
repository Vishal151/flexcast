"""Hit every Stage 0 source and print what comes back. Read-only; writes nothing.

uv run python scripts/explore_api.py [label-substring]
"""

import sys
from typing import Any

import httpx
import pandas as pd
from sources import BASE_URLS, Request, requests

OPENAPI_URL = "https://data.elexon.co.uk/swagger/v1/swagger.json"
TIME_COLS = ("startTime", "publishTime", "createdDateTime", "valid_from", "settlementDate")


def openapi_paths(client: httpx.Client) -> set[str]:
    spec = client.get(OPENAPI_URL).raise_for_status().json()
    return set(spec["paths"])


def path_in_spec(path: str, spec_paths: set[str]) -> bool:
    """Match a concrete path against templated OpenAPI paths like /x/{settlementDate}."""
    parts = path.strip("/").split("/")
    for p in spec_paths:
        tparts = p.strip("/").split("/")
        if len(tparts) == len(parts) and all(
            t == c or (t.startswith("{") and t.endswith("}"))
            for t, c in zip(tparts, parts, strict=True)
        ):
            return True
    return False


def rows_of(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, list):
        return body
    for key in ("data", "results"):
        if isinstance(body, dict) and key in body:
            rows: list[dict[str, Any]] = body[key]
            return rows
    return []


def describe(req: Request, resp: httpx.Response, spec_paths: set[str]) -> None:
    in_spec = path_in_spec(req.path, spec_paths) if req.source == "elexon" else None
    print(f"\n## {req.label}  {req.source}{req.path} {req.params}")
    print(f"   status={resp.status_code} bytes={len(resp.content)} in_openapi={in_spec}")
    if resp.status_code != 200:
        print(f"   body: {resp.text[:300]}")
        return
    body = resp.json()
    rows = rows_of(body)
    if not rows:
        print(f"   top-level keys: {list(body) if isinstance(body, dict) else type(body)}")
        return
    df = pd.DataFrame(rows)
    print(f"   rows={len(df)} cols={len(df.columns)}")
    print("   dtypes: " + ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items()))
    for col in TIME_COLS:
        if col in df:
            vals = df[col].dropna().astype(str)
            print(f"   {col}: min={vals.min()} max={vals.max()} distinct={vals.nunique()}")
    if "publishTime" in df:
        print(f"   publishTimes: {sorted(df['publishTime'].unique())[:30]}")
    for col in ("dataProvider", "fuelType", "boundary"):
        if col in df:
            print(f"   {col} counts: {df[col].value_counts().to_dict()}")
    if "settlementDate" in df and "settlementPeriod" in df:
        per_date = df.groupby("settlementDate")["settlementPeriod"].nunique().to_dict()
        print(f"   periods per date: {per_date}")


def main() -> None:
    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        spec_paths = openapi_paths(client)
        print(f"Elexon OpenAPI: {len(spec_paths)} paths")
        for req in requests():
            if needle in req.label:
                resp = client.get(BASE_URLS[req.source] + req.path, params=req.params)
                describe(req, resp, spec_paths)


if __name__ == "__main__":
    main()
