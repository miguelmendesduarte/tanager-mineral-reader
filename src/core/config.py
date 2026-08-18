"""Application settings and configuration."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    """Enumeration of log levels."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level for the application.",
    )
    log_format: str = Field(
        default="{time:DD/MM/YYYY HH:mm:ss} | {level} | {name} | {message}",
        description="Format for log messages.",
    )

    # Catalog
    catalog_base_url: str = Field(
        default="https://www.planet.com/data/stac/tanager-core-imagery",
        description="Root of Planet's open Tanager STAC catalog.",
    )
    scene_collection: str = Field(
        default="energy-mining",
        description="Collection the scene under study belongs to.",
    )
    scene_id: str = Field(
        default="20250305_053421_32_4001",
        description=(
            "Scene under study: Singrauli coalfield, Madhya Pradesh, India, "
            "acquired 2025-03-05."
        ),
    )

    # Data
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory downloaded assets are written to.",
    )
    default_assets: tuple[str, ...] = Field(
        default=(
            "ortho_radiance_hdf5",
            "ortho_sr_hdf5",
            "ortho_visual",
            "ortho_beta_udm",
        ),
        description="Assets downloaded when no explicit selection is given.",
    )

    # Downloads
    request_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Timeout in seconds for catalog and asset requests.",
    )
    download_chunk_size: int = Field(
        default=8 * 1024 * 1024,
        gt=0,
        description="Number of bytes held in memory while streaming a download.",
    )

    @property
    def item_url(self) -> str:
        """URL of the STAC item describing the scene under study.

        The catalog is a static tree, so an item is addressed by the collection
        it belongs to and its scene identifier.
        """
        root = self.catalog_base_url.rstrip("/")
        return f"{root}/{self.scene_collection}/{self.scene_id}/{self.scene_id}.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve the cached application settings.

    Call `get_settings.cache_clear()` from tests to force a fresh read.

    Returns:
        Settings: The application settings.
    """
    return Settings()
