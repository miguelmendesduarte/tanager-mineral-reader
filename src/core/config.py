"""Application settings and configuration."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import (
    AssetNotDownloadedError,
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
            "the whole mineral list. Group together any species the sensor "
            "cannot separate at its resolution, and report the group rather "
            "than guessing between them. The default set is the hydrothermal "
            "alteration assemblage at Cuprite, Nevada."
        ),
    )
    nonmineral_spectra: tuple[str, ...] = Field(
        default=(
            "Grass_AETR70_CA01-AETR-2_NPV_ASDFRa_AREF",
            "Grass_AETR95_CA01-AETR-1_NPV_ASDFRa_AREF",
            "Grass_CA01-TACA-1_meadow_NPV_ASDFRa_AREF",
        ),
        description=(
            "Spectra matched against but never reported as a mineral. Ground "
            "that is not rock still has to be recognised, because a pixel is "
            "always given the nearest reference and will take a mineral's name "
            "if nothing truer is offered. The defaults are dead vegetation, "
            "whose cellulose absorbs where carbonate does and which a "
            "greenness mask cannot see."
        ),
    )
    match_range: tuple[float, float] = Field(
        default=(2080.0, 2490.0),
        description=(
            "Wavelengths the minerals are told apart in, in nanometres. Choose "
            "it by sweeping the bounds: inside a good range the answers do not "
            "move, and a bound that cuts into an absorption both shifts its "
            "position and understates its depth. Wide enough matters as much "
            "as tight enough, since secondary absorptions often separate "
            "minerals that share a primary one."
        ),
    )

    # Masking
    vegetation_ndvi: float = Field(
        default=0.2,
        description=(
            "Pixels greener than this are left out, since plants absorb in the "
            "same shortwave region the minerals are told apart in. To choose "
            "one for a new site, sweep it and compare what each date masks: "
            "set too low it tracks sparse vegetation coming and going with the "
            "season, which makes the mask depend on when the scene was taken "
            "and manufactures disagreement between dates."
        ),
    )
    dark_reflectance: float = Field(
        default=0.05,
        gt=0,
        description=(
            "Pixels darker than this in the shortwave are left out. Shadow and "
            "water reflect too little for an absorption to stand above the "
            "noise, whatever mineral lies underneath."
        ),
    )
    red_nm: tuple[float, float] = Field(
        default=(660.0, 680.0),
        description="Wavelengths averaged as red when working out how green a pixel is.",
    )
    near_infrared_nm: tuple[float, float] = Field(
        default=(850.0, 870.0),
        description="Wavelengths averaged as near infrared for the same purpose.",
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
        """Every mineral spectrum to match against, in group order."""
        return tuple(name for names in self.mineral_groups.values() for name in names)

    @property
    def references(self) -> tuple[str, ...]:
        """Every spectrum to match against, minerals first.

        The ones that are not minerals are matched exactly like the rest. They
        earn their place by taking the pixels that would otherwise be handed a
        mineral's name for want of anything closer.
        """
        return self.species + self.nonmineral_spectra

    def group_of(self, name: str) -> str | None:
        """The group a reference spectrum is reported as.

        Args:
            name: Name of the reference spectrum.

        Returns:
            str | None: Name of its group, or None when the spectrum is not a
                mineral and so names nothing.

        Raises:
            KeyError: If the spectrum is not configured at all.
        """
        for group, names in self.mineral_groups.items():
            if name in names:
                return group
        if name in self.nonmineral_spectra:
            return None
        raise KeyError(name)

    @property
    def splib07_dir(self) -> Path:
        """Directory the spectral library is downloaded to and unpacked in."""
        return self.data_dir / "splib07"

    @property
    def splib07_archive(self) -> Path:
        """Local path of the spectral library archive."""
        return self.splib07_dir / self.splib07_archive_name

    def scene_asset(self, scene_id: str, asset: str) -> Path:
        """Local path of an asset already downloaded for a scene.

        The extension varies by asset, so the file is looked up by name rather
        than assumed.

        Args:
            scene_id: Identifier of the scene.
            asset: Catalog key of the asset, e.g. `ortho_sr_hdf5`.

        Returns:
            Path: Location of the downloaded file.

        Raises:
            AssetNotDownloadedError: If no such file is present.
        """
        directory = self.data_dir / scene_id
        found = sorted(directory.glob(f"*_{asset}.*")) if directory.is_dir() else []
        if not found:
            raise AssetNotDownloadedError(scene_id, asset, directory)
        return found[0]

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
