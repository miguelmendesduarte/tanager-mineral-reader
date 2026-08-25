"""Writing a mineral map as a GeoTIFF.

A result that only exists inside Python cannot be checked by anyone. Written
out with its projection attached, the map opens in QGIS or ArcGIS beside a
published map of the same ground, which is the only way the work gets argued
with rather than taken on trust.

The class layer alone would hide how each pixel was decided, so the numbers
behind it travel in the same file: how far the pixel was from its reference,
how deep its absorption was, how much it beat the runner-up by, and whether
that margin was larger than noise could account for.
"""

from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from rasterio.transform import Affine

from ..readers.grid import Grid
from ..spectra.mapping import UNCLASSIFIED, Mapped

BANDS = ("class", "angle", "depth", "margin", "settled")


def write_map(mapped: Mapped, grid: Grid, path: Path) -> Path:
    """Write a mineral map and the numbers behind it to a GeoTIFF.

    Args:
        mapped: The map to write.
        grid: Where the scene sits on the map.
        path: File to write. Parent directories are created.

    Returns:
        Path: The file written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = (
        mapped.labels.astype(np.float32),
        mapped.angle.astype(np.float32),
        mapped.depth.astype(np.float32),
        mapped.margin.astype(np.float32),
        mapped.resolved.astype(np.float32),
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=mapped.labels.shape[0],
        width=mapped.labels.shape[1],
        count=len(BANDS),
        dtype="float32",
        crs=rasterio.crs.CRS.from_epsg(grid.epsg),
        transform=Affine.from_gdal(*grid.transform),
        nodata=float(UNCLASSIFIED),
        compress="deflate",
        tiled=True,
    ) as raster:
        for index, (name, layer) in enumerate(zip(BANDS, layers, strict=True), start=1):
            raster.write(layer, index)
            raster.set_band_description(index, name)
        raster.update_tags(**_tags(mapped))

    logger.info("Wrote {} ({:.1f} MB)", path, path.stat().st_size / 1024 / 1024)
    return path


def _tags(mapped: Mapped) -> dict[str, str]:
    """What the file needs to carry to be readable without this code.

    The class numbers mean nothing on their own, and neither does a map whose
    thresholds are unrecorded, so both travel with the pixels.
    """
    classes = {str(index): group for index, group in enumerate(mapped.groups)} | {
        str(UNCLASSIFIED): "unclassified"
    }
    return {
        "classes": "; ".join(f"{value}={name}" for value, name in classes.items()),
        "noise_reflectance": f"{mapped.floor.noise:.6f}",
        "depth_floor": f"{mapped.floor.depth:.4f}",
        "angle_floor": f"{mapped.floor.angle:.2f}",
        "floor_quantile": f"{mapped.floor.quantile}",
        "resolution_quantile": f"{mapped.resolution.quantile}",
        "attribution": (
            "Tanager STAC Data, available at www.planet.com/data/stac "
            "(c) Planet Labs PBC. All Rights Reserved. Reference spectra from "
            "the USGS Spectral Library Version 7 (Kokaly et al., 2017, "
            "USGS Data Series 1035)."
        ),
    }
