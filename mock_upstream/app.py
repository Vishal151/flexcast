"""A stand-in for Elexon and Octopus that replays tests/fixtures by manifest.

Routes are /<source><path>, e.g. /elexon/datasets/NDF?..., so the app's base URL settings become
http://upstream-mock:8080/elexon and .../octopus. Only exact (path, query) matches are served;
anything else is a 404, so a client asking for the wrong thing fails loudly.
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

Key = tuple[str, frozenset[tuple[str, str]]]


def create_app(fixtures: Path) -> FastAPI:
    manifest = json.loads((fixtures / "manifest.json").read_text())
    routes: dict[Key, Path] = {
        (f"/{e['source']}{e['path']}", frozenset(e["params"].items())): fixtures / e["file"]
        for e in manifest["entries"]
    }
    app = FastAPI(title="flexcast upstream mock", redirect_slashes=False)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "fixtures": len(routes)}

    @app.get("/{path:path}")
    def replay(path: str, request: Request) -> Response:
        key = (f"/{path}", frozenset(request.query_params.items()))
        if key not in routes:
            return JSONResponse({"error": "no fixture", "path": key[0]}, status_code=404)
        return Response(routes[key].read_bytes(), media_type="application/json")

    return app


app = create_app(Path(os.environ.get("MOCK_FIXTURES", "tests/fixtures")))
