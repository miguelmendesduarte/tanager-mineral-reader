"""Reading Tanager hyperspectral cubes.

The radiance and the surface reflectance products are HDF-EOS5 files with the
same layout: one cube of shape (band, row, column) beside a handful of
two dimensional masks, all under a single grid. Only the name of the cube
differs, so one reader serves both.

The whole cube is a little under a gigabyte, so reads are deliberately partial:
a selection of bands, a spatial window, or a single spectrum.
"""

from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import LayerNotFoundError, UnknownCubeError
from .grid import Grid
from .wavelengths import Wavelengths

GRID_PATH = "HDFEOS/GRIDS/HYP"
FIELDS_PATH = f"{GRID_PATH}/Data Fields"
STRUCT_METADATA_PATH = "HDFEOS INFORMATION/StructMetadata.0"

CUBE_NAMES = ("toa_radiance", "surface_reflectance")
UNCERTAINTY = "surface_reflectance_uncertainty"
EXCLUDING_MASKS = ("nodata_pixels", "beta_cloud_mask", "beta_cirrus_mask")


class Cube:
    """A hyperspectral cube together with its spectral and map metadata.

    Use it as a context manager so that the underlying file is closed:

        with Cube(path) as cube:
            bands = cube.bands_between(2100, 2450)
            window = cube.read_bands(bands)
    """

    def __init__(self, path: Path) -> None:
        """Open a cube.

        Args:
            path: Path to the HDF-EOS5 file.

        Raises:
            UnknownCubeError: If the file holds no cube this project reads.
        """
        self._path = path
        self._file = h5py.File(path, "r")

        available = list(self._file[FIELDS_PATH])
        found = [name for name in CUBE_NAMES if name in available]
        if not found:
            self._file.close()
            raise UnknownCubeError(path, available)
        self._name = found[0]

    @property
    def name(self) -> str:
        """Name of the cube held by this file."""
        return self._name

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of bands, rows and columns."""
        bands, rows, columns = self._cube.shape
        return int(bands), int(rows), int(columns)

    @property
    def fill_value(self) -> float:
        """Value the file uses for pixels that hold no measurement."""
        return float(self._cube.attrs["_FillValue"])

    @cached_property
    def wavelengths(self) -> Wavelengths:
        """The spectral axis of the cube."""
        attributes = self._cube.attrs
        centres = np.asarray(attributes["wavelengths"], dtype=np.float64)
        flags = attributes.get("good_wavelengths")
        usable = (
            np.ones(centres.size, dtype=np.bool_)
            if flags is None
            else np.asarray(flags, dtype=np.bool_)
        )
        return Wavelengths(
            centres=centres,
            widths=np.asarray(attributes["fwhm"], dtype=np.float64),
            usable=usable,
        )

    @property
    def strip_id(self) -> str | None:
        """Identifier of the pass the scene was cut from.

        Tanager records a long strip and delivers it in pieces, so two scenes
        can share everything except the fact that they are one observation.
        Comparing two pieces of one strip measures nothing, and only this
        attribute tells them apart — the catalog does not carry it.

        Returns:
            str | None: The identifier, or None if the product omits it.
        """
        value = self._file[GRID_PATH].attrs.get("strip_id")
        if value is None:
            return None
        return str(value.decode() if isinstance(value, bytes) else value)

    @cached_property
    def grid(self) -> Grid:
        """Where the scene sits on the map."""
        _, rows, columns = self.shape
        metadata = self._file[STRUCT_METADATA_PATH][()]
        return Grid.from_metadata(
            metadata.decode("utf-8", errors="ignore"),
            epsg=int(self._file[GRID_PATH].attrs["epsg_code"]),
            shape=(rows, columns),
        )

    def bands_between(
        self,
        low: float,
        high: float,
        *,
        usable_only: bool = True,
    ) -> NDArray[np.intp]:
        """Indices of the bands inside a wavelength range, in nanometres.

        Args:
            low: Lower bound in nanometres, included.
            high: Upper bound in nanometres, included.
            usable_only: Skip the bands the file flags as unusable.

        Returns:
            NDArray[np.intp]: Indices of the selected bands, in order.

        Raises:
            NoBandsInRangeError: If the range holds no selectable band.
        """
        return self.wavelengths.indices_between(low, high, usable_only=usable_only)

    def read_bands(self, indices: NDArray[np.intp]) -> NDArray[np.float32]:
        """Read whole bands.

        Args:
            indices: Indices of the bands to read. Duplicates are dropped and
                the result follows increasing band order.

        Returns:
            NDArray[np.float32]: Cube of shape (band, row, column), with fill
                values replaced by NaN.
        """
        wanted = np.unique(np.asarray(indices, dtype=np.intp))
        return self._without_fill(self._cube[wanted, :, :])

    def read_window(
        self,
        rows: slice,
        columns: slice,
        *,
        bands: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float32]:
        """Read a rectangular part of the scene.

        Args:
            rows: Rows to read.
            columns: Columns to read.
            bands: Bands to read; every band when omitted.

        Returns:
            NDArray[np.float32]: Cube of shape (band, row, column), with fill
                values replaced by NaN.
        """
        selection = (
            slice(None)
            if bands is None
            else np.unique(np.asarray(bands, dtype=np.intp))
        )
        return self._without_fill(self._cube[selection, rows, columns])

    def read_spectrum(self, row: int, column: int) -> NDArray[np.float32]:
        """Read every band of a single pixel.

        Args:
            row: Row of the pixel.
            column: Column of the pixel.

        Returns:
            NDArray[np.float32]: One value per band, with fill values replaced
                by NaN.
        """
        return self._without_fill(self._cube[:, row, column])

    @property
    def layers(self) -> tuple[str, ...]:
        """Names of the per pixel layers this file carries beside the cube.

        Depending on the product these cover the viewing and illumination
        geometry, the masks, and the atmosphere the correction estimated.
        """
        return tuple(
            sorted(name for name in self._file[FIELDS_PATH] if name != self._name)
        )

    def read_layer(self, name: str) -> NDArray[np.float32]:
        """Read one per pixel layer.

        Args:
            name: Name of the layer, as listed by `layers`.

        Returns:
            NDArray[np.float32]: Layer of shape (row, column), with fill values
                replaced by NaN.

        Raises:
            LayerNotFoundError: If the file carries no such layer.
        """
        fields = self._file[FIELDS_PATH]
        if name not in self.layers:
            raise LayerNotFoundError(name, self.layers)

        layer = fields[name]
        values = np.asarray(layer[:], dtype=np.float32)
        fill = layer.attrs.get("_FillValue")
        if fill is None:
            return values
        return np.where(values == np.float32(fill), np.float32(np.nan), values)

    def noise(self, indices: NDArray[np.intp], *, step: int = 1) -> float:
        """Typical reflectance uncertainty over a selection of bands.

        The product carries its own estimate of how far each measurement could
        be out, which is what makes it possible to ask whether an absorption is
        real without picking a threshold. The layer is as large as the cube, so
        it is sampled rather than read whole.

        Args:
            indices: Bands to measure over.
            step: Read every `step`th row and column.

        Returns:
            float: Median uncertainty, in reflectance.

        Raises:
            LayerNotFoundError: If the product carries no uncertainty layer.
        """
        fields = self._file[FIELDS_PATH]
        if UNCERTAINTY not in fields:
            raise LayerNotFoundError(UNCERTAINTY, self.layers)

        wanted = np.unique(np.asarray(indices, dtype=np.intp))
        values = fields[UNCERTAINTY][wanted[0] : wanted[-1] + 1, ::step, ::step]
        return float(np.nanmedian(self._without_fill(values)))

    def valid_mask(self) -> NDArray[np.bool_]:
        """Pixels that hold a usable measurement.

        A pixel is usable when it falls inside the imaged strip and carries
        neither cloud nor cirrus.

        Returns:
            NDArray[np.bool_]: Mask of shape (row, column).
        """
        fields = self._file[FIELDS_PATH]
        _, rows, columns = self.shape
        valid = np.ones((rows, columns), dtype=np.bool_)
        for name in EXCLUDING_MASKS:
            if name in fields:
                valid &= np.asarray(fields[name][:]) == 0
        return valid

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    def __enter__(self) -> "Cube":
        """Enter a context that closes the file on exit."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying file."""
        self.close()

    @property
    def _cube(self) -> Any:
        """The cube dataset. Typed loosely because h5py ships no annotations."""
        return self._file[f"{FIELDS_PATH}/{self._name}"]

    def _without_fill(self, values: NDArray[Any]) -> NDArray[np.float32]:
        """Replace fill values with NaN, so that gaps cannot pass unnoticed."""
        array = np.asarray(values, dtype=np.float32)
        return np.where(array == np.float32(self.fill_value), np.float32(np.nan), array)
