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
    scene_ids: tuple[str, ...] = Field(
        default=(
            "20240925_185504_87_4001",
            "20250222_190237_16_4001",
        ),
        min_length=1,
        description=(
            "Scenes under study: Cuprite, Nevada, acquired 2024-09-25 and "
            "2025-02-22. Two separate collects over the same ground, which is "
            "what makes them worth comparing."
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

    # Spectral library
    sciencebase_base_url: str = Field(
        default="https://www.sciencebase.gov/catalog",
        description="Root of the USGS ScienceBase catalog.",
    )
    splib07_item_id: str = Field(
        default="5807a2a2e4b0841e59e3a18d",
        description="USGS Spectral Library Version 7, a single 5.5 GB archive.",
    )
    splib07_archive_name: str = Field(
        default="usgs_splib07.zip",
        description="Name of the archive within the catalog record.",
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
    def splib07_dir(self) -> Path:
        """Directory the spectral library is downloaded to and unpacked in."""
        return self.data_dir / "splib07"

    @property
    def splib07_item_url(self) -> str:
        """URL of the catalog record describing the spectral library."""
        root = self.sciencebase_base_url.rstrip("/")
        return f"{root}/item/{self.splib07_item_id}?format=json"

    @property
    def splib07_url(self) -> str:
        """URL the spectral library archive is downloaded from."""
        root = self.sciencebase_base_url.rstrip("/")
        return (
            f"{root}/file/get/{self.splib07_item_id}?name={self.splib07_archive_name}"
        )

    def item_url(self, scene_id: str) -> str:
        """URL of the STAC item describing one scene.

        The catalog is a static tree, so an item is addressed by the collection
        it belongs to and its scene identifier.

        Args:
            scene_id: Identifier of the scene, e.g. `20250222_190237_16_4001`.

        Returns:
            str: URL of the item JSON.
        """
        root = self.catalog_base_url.rstrip("/")
        return f"{root}/{self.scene_collection}/{scene_id}/{scene_id}.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve the cached application settings.

    Call `get_settings.cache_clear()` from tests to force a fresh read.

    Returns:
        Settings: The application settings.
    """
    return Settings()
