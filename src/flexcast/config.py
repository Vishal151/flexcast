"""Runtime configuration. Everything comes from env vars (prefix FLEXCAST_); profiles differ only
in these values, never in code paths."""

from datetime import time
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLEXCAST_")

    data_uri: str = "file://./data"
    elexon_base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"
    octopus_base_url: str = "https://api.octopus.energy/v1"
    neso_base_url: str = "https://api.neso.energy/api/3/action"
    agile_region: str = "C"
    http_concurrency: int = 4
    git_sha: str = "unknown"  # recorded in model metadata; set by make / CI
    decision_at: time = time(9)  # D-1 wall-clock time, Europe/London
    # The market the battery trades (SPEC §1): the day-ahead auction; imbalance is the comparison.
    market: Literal["imbalance", "day_ahead"] = "day_ahead"
    risk_alpha: float = 0.0  # chosen by the backtest's validation window (docs/results.md)
    # Rolling conformal calibration of the quantiles, chosen in notebook 05: 90 days, one offset
    # for the whole day, band only (p50 and so the schedule unchanged). 0 days = off.
    calibration_window_days: int = 90
    calibration_hours_per_block: int | None = None
    calibration_adjust: str = "p10,p90"
    # Cloud Run injects PORT (unprefixed); FLEXCAST_PORT also works.
    port: int = Field(default=8080, validation_alias=AliasChoices("PORT", "FLEXCAST_PORT"))
