"""Runtime configuration. Everything comes from env vars (prefix FLEXCAST_); profiles differ only
in these values, never in code paths."""

from datetime import time

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLEXCAST_")

    data_uri: str = "file://./data"
    elexon_base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"
    octopus_base_url: str = "https://api.octopus.energy/v1"
    agile_region: str = "C"
    http_concurrency: int = 4
    git_sha: str = "unknown"  # recorded in model metadata; set by make / CI
    decision_at: time = time(9)  # D-1 wall-clock time, Europe/London
    risk_alpha: float = 0.0  # chosen by the backtest's validation window (docs/results.md)
    # Cloud Run injects PORT (unprefixed); FLEXCAST_PORT also works.
    port: int = Field(default=8080, validation_alias=AliasChoices("PORT", "FLEXCAST_PORT"))
