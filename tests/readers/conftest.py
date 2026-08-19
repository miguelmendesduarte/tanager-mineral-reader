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
