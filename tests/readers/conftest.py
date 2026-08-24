"""Synthetic cubes that mirror the layout of a real Tanager file."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.readers.cube import FIELDS_PATH, GRID_PATH, STRUCT_METADATA_PATH

BANDS = 6
ROWS = 4
COLUMNS = 5
FILL_VALUE = -9999.0
EPSG = 32644
UPPER_LEFT = (655410.0, 2681190.0)
LOWER_RIGHT = (655560.0, 2681070.0)

STRUCT_METADATA = f"""GROUP=GridStructure
\tGROUP=GRID_1
\t\tGridName="HYP"
\t\tUpperLeftPointMtrs=({UPPER_LEFT[0]:.2f},{UPPER_LEFT[1]:.2f})
\t\tLowerRightMtrs=({LOWER_RIGHT[0]:.2f},{LOWER_RIGHT[1]:.2f})
\t\tProjection=HE5_GCTP_UTM
\tEND_GROUP=GRID_1
END_GROUP=GridStructure
"""


def write_cube(
    path: Path,
    *,
    name: str = "surface_reflectance",
    usable: list[int] | None = None,
    masks: dict[str, np.ndarray] | None = None,
    layer_fill: float | None = None,
    struct_metadata: str = STRUCT_METADATA,
) -> Path:
    """Write a small file shaped like a real cube.

    Band `b`, row `r` and column `c` holds the value `b * 100 + r * 10 + c`,
    which makes it easy to assert that the right part of the cube was read.
    The first pixel of the first band holds the fill value instead.
    """
    bands = np.arange(BANDS)[:, None, None] * 100
    rows = np.arange(ROWS)[None, :, None] * 10
    columns = np.arange(COLUMNS)[None, None, :]
    values = (bands + rows + columns).astype(np.float32)
    values[0, 0, 0] = FILL_VALUE

    with h5py.File(path, "w") as file:
        fields = file.create_group(FIELDS_PATH)
        cube = fields.create_dataset(name, data=values)
        cube.attrs["wavelengths"] = np.linspace(400.0, 2500.0, BANDS)
        cube.attrs["fwhm"] = np.full(BANDS, 5.0)
        cube.attrs["_FillValue"] = FILL_VALUE
        if usable is not None:
            cube.attrs["good_wavelengths"] = np.asarray(usable, dtype=np.uint8)

        for layer_name, layer in (masks or {}).items():
            dataset = fields.create_dataset(layer_name, data=layer.astype(np.float32))
            if layer_fill is not None:
                dataset.attrs["_FillValue"] = layer_fill

        file[GRID_PATH].attrs["epsg_code"] = EPSG
        file.create_dataset(STRUCT_METADATA_PATH, data=np.bytes_(struct_metadata))

    return path


@pytest.fixture
def cube_path(tmp_path: Path) -> Path:
    """Path to a cube with no masks and every band usable."""
    return write_cube(tmp_path / "cube.h5")


SPECTRAL_BANDS = {"red": 670.0, "near_infrared": 860.0, "shortwave": 2200.0}
SPECTRAL_DEFAULTS = {"red": 0.20, "near_infrared": 0.25, "shortwave": 0.30}


def write_spectral_cube(
    path: Path,
    *,
    red: list[float] | None = None,
    near_infrared: list[float] | None = None,
    shortwave: list[float] | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a cube with bands where greenness and brightness are measured.

    Each argument gives the reflectance of the first row's leading pixels in
    that band; every other pixel takes a default that reads as bare, lit rock.
    """
    given = {"red": red, "near_infrared": near_infrared, "shortwave": shortwave}
    values = np.empty((len(SPECTRAL_BANDS), ROWS, COLUMNS), dtype=np.float32)

    for band, name in enumerate(SPECTRAL_BANDS):
        values[band] = SPECTRAL_DEFAULTS[name]
        for column, value in enumerate(given[name] or []):
            values[band, 0, column] = value

    with h5py.File(path, "w") as file:
        fields = file.create_group(FIELDS_PATH)
        cube = fields.create_dataset("surface_reflectance", data=values)
        cube.attrs["wavelengths"] = np.array(list(SPECTRAL_BANDS.values()))
        cube.attrs["fwhm"] = np.full(len(SPECTRAL_BANDS), 5.5)
        cube.attrs["_FillValue"] = FILL_VALUE

        for layer_name, layer in (masks or {}).items():
            fields.create_dataset(layer_name, data=layer.astype(np.float32))

        file[GRID_PATH].attrs["epsg_code"] = EPSG
        file.create_dataset(STRUCT_METADATA_PATH, data=np.bytes_(STRUCT_METADATA))

    return path
