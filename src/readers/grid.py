"""The map grid a scene is projected onto."""

import re
from dataclasses import dataclass

from ..core.exceptions import (
    DifferentPixelSizesError,
    DifferentProjectionsError,
    GridMetadataError,
    OffLatticeError,
)

UPPER_LEFT = "UpperLeftPointMtrs"
LOWER_RIGHT = "LowerRightMtrs"


@dataclass(frozen=True)
class Grid:
    """Where a scene sits on the map, in projected metres.

    Attributes:
        epsg: Code of the projected coordinate system.
        upper_left: Easting and northing of the top left corner.
        lower_right: Easting and northing of the bottom right corner.
        shape: Number of rows and columns.
    """

    epsg: int
    upper_left: tuple[float, float]
    lower_right: tuple[float, float]
    shape: tuple[int, int]

    @classmethod
    def from_metadata(
        cls,
        metadata: str,
        *,
        epsg: int,
        shape: tuple[int, int],
    ) -> "Grid":
        """Read a grid from the structure metadata carried by the file.

        Args:
            metadata: Contents of the HDF-EOS structure metadata block.
            epsg: Code of the projected coordinate system.
            shape: Number of rows and columns of the cube.

        Returns:
            Grid: The grid described by the metadata.

        Raises:
            GridMetadataError: If a corner is missing from the metadata.
        """
        return cls(
            epsg=epsg,
            upper_left=_corner(metadata, UPPER_LEFT),
            lower_right=_corner(metadata, LOWER_RIGHT),
            shape=shape,
        )

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Width and height of a pixel in metres, both positive."""
        rows, columns = self.shape
        left, top = self.upper_left
        right, bottom = self.lower_right
        return (right - left) / columns, (top - bottom) / rows

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """Affine transform from pixel to map coordinates, in GDAL order."""
        width, height = self.pixel_size
        left, top = self.upper_left
        return left, width, 0.0, top, 0.0, -height

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Extent of the scene as left, bottom, right and top."""
        left, top = self.upper_left
        right, bottom = self.lower_right
        return left, bottom, right, top

    def offset_to(self, other: "Grid") -> tuple[int, int]:
        """Where another grid's first pixel sits inside this one.

        Two scenes can only be compared pixel to pixel if they fall on the same
        lattice: same projection, same pixel size, and origins a whole number of
        pixels apart. When they do, comparing them is a matter of taking a
        window out of the larger one, and no resampling is involved at all.

        Args:
            other: Grid to locate inside this one.

        Returns:
            tuple[int, int]: Row and column of this grid at which the other
                grid begins. Either may be negative if it starts outside.

        Raises:
            GridsNotAlignedError: If the two grids do not share a lattice.
                Resampling one onto the other is deliberately not implemented,
                so this is a refusal rather than a fallback.
        """
        if self.epsg != other.epsg:
            raise DifferentProjectionsError(self.epsg, other.epsg)

        if self.pixel_size != other.pixel_size:
            raise DifferentPixelSizesError(self.pixel_size, other.pixel_size)

        width, height = self.pixel_size
        columns = (other.upper_left[0] - self.upper_left[0]) / width
        rows = (self.upper_left[1] - other.upper_left[1]) / height
        if not (_is_whole(rows) and _is_whole(columns)):
            raise OffLatticeError(rows, columns)

        return round(rows), round(columns)


def _corner(metadata: str, name: str) -> tuple[float, float]:
    """Read one corner of the grid from the structure metadata."""
    match = re.search(rf"{name}=\((-?[\d.]+),(-?[\d.]+)\)", metadata)
    if match is None:
        raise GridMetadataError(name)
    return float(match.group(1)), float(match.group(2))


def _is_whole(value: float, tolerance: float = 1e-6) -> bool:
    """Whether a number of pixels is whole, allowing for floating point."""
    return abs(value - round(value)) <= tolerance
