"""Models for the subset of STAC that this project consumes.

The catalog carries far more metadata than we need, so the models below are
deliberately permissive: unknown fields are ignored rather than rejected, which
keeps the pipeline working if Planet extends the schema.
"""

from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from ..core.exceptions import AssetNotFoundError


class Asset(BaseModel):
    """A single downloadable file attached to an item."""

    model_config = ConfigDict(extra="ignore")

    href: str
    type: str | None = None
    roles: tuple[str, ...] = ()
    description: str | None = None

    @property
    def filename(self) -> str:
        """Name of the file as published in the catalog.

        Only the path is considered, so that a signed href keeps its real name
        instead of dragging the query string into the filename.
        """
        return urlsplit(self.href).path.rsplit("/", 1)[-1]


class Item(BaseModel):
    """A single acquisition and the assets derived from it."""

    model_config = ConfigDict(extra="ignore")

    id: str
    bbox: tuple[float, float, float, float] | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    assets: dict[str, Asset] = Field(default_factory=dict)

    @property
    def acquired_at(self) -> str | None:
        """Acquisition timestamp, as published by the catalog."""
        return self._property("datetime")

    @property
    def location(self) -> str | None:
        """Human readable description of the imaged area."""
        return self._property("location_description")

    def asset(self, name: str) -> Asset:
        """Look up an asset by its catalog key.

        Args:
            name: Key of the asset, e.g. `ortho_radiance_hdf5`.

        Returns:
            Asset: The requested asset.

        Raises:
            AssetNotFoundError: If the item does not expose that asset.
        """
        try:
            return self.assets[name]
        except KeyError:
            raise AssetNotFoundError(name, self.assets) from None

    def _property(self, name: str) -> str | None:
        """Read a string property, returning None when it is absent."""
        value = self.properties.get(name)
        return None if value is None else str(value)
