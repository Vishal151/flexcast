"""HTTP API over stored plans. Read-only: plans are produced by the `plan` job."""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from flexcast.config import Settings
from flexcast.history import History
from flexcast.jobs import LATEST, plan_metrics
from flexcast.settlement import LONDON
from flexcast.storage import Storage


class Health(BaseModel):
    status: str
    latest_model: str | None


class SchedulePeriod(BaseModel):
    start_time: AwareDatetime
    charge_kw: float
    discharge_kw: float
    soc_kwh: float


class ForecastPeriod(BaseModel):
    start_time: AwareDatetime
    p10: float
    p50: float
    p90: float


class PlanInfo(BaseModel):
    target_date: date
    decision_time: AwareDatetime
    model: str
    risk_alpha: float
    expected_gbp: float


class ScheduleOut(PlanInfo):
    periods: list[SchedulePeriod]


class ForecastOut(PlanInfo):
    periods: list[ForecastPeriod]


class Metrics(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    days: int
    from_: date | None = Field(alias="from")
    to: date | None
    mae: float | None = Field(description="Mean absolute error of p50, £/MWh")
    settled_gbp: float = Field(description="£ the planned schedules earned at actual prices")
    perfect_gbp: float = Field(description="£ with perfect foresight over the same days")
    pct_of_perfect: float | None


DateParam = Annotated[date | None, Query(description="Target London day; default tomorrow")]


def _tomorrow() -> date:
    return datetime.now(UTC).astimezone(LONDON).date() + timedelta(days=1)


def create_app(
    storage: Storage | None = None, load_history: Callable[[], History] | None = None
) -> FastAPI:
    store = storage or Storage(Settings().data_uri)
    history = load_history or (lambda: History.load(store))
    app = FastAPI(title="flexcast", description="Battery schedules from GB price forecasts.")

    def read_plan(d: date, table: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        prefix = f"plans/date={d}"
        if not store.exists(f"{prefix}/plan.json"):
            raise HTTPException(404, f"no plan for {d}")
        meta = json.loads(store.read_text(f"{prefix}/plan.json"))
        frame = store.read_parquet(f"{prefix}/{table}.parquet")
        rows = [{str(k): v for k, v in r.items()} for r in frame.to_dict("records")]
        return meta, rows

    @app.get("/health")
    def health() -> Health:
        try:
            latest = store.read_text(LATEST).strip() if store.exists(LATEST) else None
        except Exception as e:  # storage unreachable: unhealthy, not a 500 with a stack trace
            raise HTTPException(503, f"storage unavailable: {e}") from e
        return Health(status="ok", latest_model=latest)

    @app.get("/schedule")
    def schedule(date: DateParam = None) -> ScheduleOut:
        meta, rows = read_plan(date or _tomorrow(), "schedule")
        return ScheduleOut.model_validate({**meta, "periods": rows})

    @app.get("/forecast")
    def forecast(date: DateParam = None) -> ForecastOut:
        meta, rows = read_plan(date or _tomorrow(), "forecast")
        return ForecastOut.model_validate({**meta, "periods": rows})

    @app.get("/metrics")
    def metrics(
        end: Annotated[date | None, Query(description="Last target day; default today")] = None,
        days: Annotated[int, Query(ge=1, le=366)] = 30,
    ) -> Metrics:
        last = end or datetime.now(UTC).astimezone(LONDON).date()
        return Metrics.model_validate(plan_metrics(history(), store, last, days))

    return app
