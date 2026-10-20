# Optional dev UI over the same compose stack and make targets. Nothing in the project needs
# Tilt: every button below runs a `make` target, and CI never uses this file.
#   tilt up      -> http://localhost:10350
#   tilt down
# Use either Tilt or `make up`/`make down` for the stack, not both at once (same compose project).

docker_compose("compose.yaml")

# Rebuild flexcast:local (used by api, upstream-mock, gcs-init and the jobs) when its inputs change.
docker_build("flexcast:local", ".", only=["pyproject.toml", "uv.lock", "src"])

dc_resource("gcs", labels=["infra"])
dc_resource("gcs-init", labels=["infra"])
dc_resource("upstream-mock", labels=["infra"], links=[link("http://localhost:8081/health", "health")])
dc_resource("api", labels=["app"], links=[link("http://localhost:8080/docs", "API docs")])

# Checks. Unit tests re-run on save; the slower ones are buttons.
local_resource("unit tests", "make test", deps=["src", "tests/unit"], labels=["checks"])
local_resource("lint", "make lint", auto_init=False, trigger_mode=TRIGGER_MODE_MANUAL, labels=["checks"])
local_resource(
    "integration tests",
    "make itest",
    resource_deps=["api", "upstream-mock"],
    auto_init=False,
    trigger_mode=TRIGGER_MODE_MANUAL,
    labels=["checks"],
)

# Load fixture data, train and plan inside the stack (same jobs Cloud Run will run).
local_resource(
    "demo pipeline",
    "make demo",
    resource_deps=["api", "upstream-mock"],
    auto_init=False,
    trigger_mode=TRIGGER_MODE_MANUAL,
    labels=["jobs"],
)
