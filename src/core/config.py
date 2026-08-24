"""Application settings and configuration."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import (
    EmptyMatchRangeError,
    EmptyMineralGroupError,
    NoMineralGroupsError,
    RepeatedSpeciesError,
)


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

    # Minerals
    mineral_groups: dict[str, tuple[str, ...]] = Field(
        default={
            "alunite": ("Alunite_HS295.3B_ASDNGa_AREF",),
            "kaolinite_group": (
                "Kaolinite_KGa-1_(wxl)_ASDNGb_AREF",
                "Dickite_NMNH106242_ASDNGb_AREF",
                "Halloysite_NMNH106237_ASDNGa_AREF",
            ),
            "muscovite": ("Muscovite_GDS113_Ruby_ASDNGa_AREF",),
            "montmorillonite": ("Montmorillonite_SWy-1_ASDNGb_AREF",),
            "pyrophyllite": ("Pyrophyllite_PYS1A_lt850um_ASDNGa_AREF",),
            "carbonate": ("Calcite_WS272_ASDNGa_AREF",),
        },
        description=(
            "Reference spectra to match pixels against, under the group each "
            "is reported as. A species appears exactly once, so the groups are "
            "the whole mineral list. Which minerals these are, and why, is "
            "recorded in docs/decisions.md 018 and 019."
        ),
    )
    match_range: tuple[float, float] = Field(
        default=(2080.0, 2400.0),
        description=(
            "Wavelengths the minerals are told apart in, in nanometres. This "
            "is where their Al-OH, Mg-OH and carbonate absorptions fall, and "
            "keeping it tight keeps the continuum local to the features being "
            "compared."
        ),
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

    @field_validator("mineral_groups")
    @classmethod
    def _reject_empty_or_repeated(
        cls,
        groups: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        """Keep the groups a single source of truth for the mineral list.

        A species under two groups would have no one answer to what it is
        reported as, and an empty group would name a class no pixel can take.
        """
        if not groups:
            raise NoMineralGroupsError

        seen: set[str] = set()
        for group, species in groups.items():
            if not species:
                raise EmptyMineralGroupError(group)
            repeated = seen & set(species)
            if repeated:
                raise RepeatedSpeciesError(repeated)
            seen |= set(species)

        return groups

    @field_validator("match_range")
    @classmethod
    def _reject_backwards_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Refuse a range that ends before it starts."""
        low, high = value
        if low >= high:
            raise EmptyMatchRangeError(low, high)
        return value

    @property
    def species(self) -> tuple[str, ...]:
        """Every reference spectrum to match against, in group order."""
        return tuple(name for names in self.mineral_groups.values() for name in names)

    def group_of(self, name: str) -> str:
        """The group a reference spectrum is reported as.

        Args:
            name: Name of the reference spectrum.

        Returns:
            str: Name of its group.

        Raises:
            KeyError: If no group holds that spectrum.
        """
        for group, names in self.mineral_groups.items():
            if name in names:
                return group
        raise KeyError(name)

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
