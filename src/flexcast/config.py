"""Runtime configuration. Everything comes from env vars (prefix FLEXCAST_); profiles differ only
in these values, never in code paths."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLEXCAST_")

    data_uri: str = "file://./data"
    elexon_base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"
    octopus_base_url: str = "https://api.octopus.energy/v1"
    agile_region: str = "C"
    http_concurrency: int = 4
