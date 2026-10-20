# syntax=docker/dockerfile:1.7
# One image, many entrypoints: `flexcast ingest|train|plan|serve|backtest`.

# --- builder: resolve the virtualenv with uv (build tools stay in this stage) -----------------
FROM ghcr.io/astral-sh/uv:0.11.17 AS uv
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first, from the lockfile only: this layer is reused until uv.lock changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-default-groups --no-install-project

# Then the project itself (changes often, rebuilds in seconds). Non-editable: the venv holds a
# real copy of the package, so the runtime stage needs nothing but /app/.venv.
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable \
    # Third-party test suites (pandas, scipy, pyarrow...) never run here: ~135 MB.
    && find /app/.venv -type d -name tests -prune -exec rm -rf {} +

# --- runtime: slim base, native libs only, non-root -------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
# libgomp1: OpenMP runtime LightGBM links against (libomp on macOS).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 flexcast

COPY --from=builder --chown=flexcast:flexcast /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8080 \
    # gcsfs's experimental zonal/HNS client probes buckets over gRPC; plain buckets don't need it
    # and the emulator can't answer. Off in every profile so local and GCP run the same client.
    GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT=false

USER flexcast
WORKDIR /home/flexcast
EXPOSE 8080
ENTRYPOINT ["flexcast"]
CMD ["serve"]
