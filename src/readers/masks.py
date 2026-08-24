"""Deciding which pixels of a scene could be bare rock.

Matching a pixel against a mineral only means something where the ground is
actually exposed. Three things have to go: pixels the file itself flags as
unusable, pixels covered by plants, and pixels too dark for an absorption to
be measurable.

The first comes from the file. The other two are thresholds, and thresholds
invite arguments, so `rock` reports how many pixels each one removed: the
effect of a choice should be visible rather than assumed, and a threshold that
removes half a scene should be obvious immediately.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .cube import Cube


def greenness(
    cube: Cube,
    red: tuple[float, float],
    near_infrared: tuple[float, float],
) -> NDArray[np.float64]:
    """How green each pixel is, as a normalised difference.

    Leaves absorb red light and reflect near infrared strongly, so the
    difference between the two bands separates plants from rock and soil far
    better than either band alone.

    Args:
        cube: Scene to measure.
        red: Wavelengths to average as red, in nanometres.
        near_infrared: Wavelengths to average as near infrared.

    Returns:
        NDArray[np.float64]: One value per pixel, from -1 to 1. Pixels where
            the two bands sum to zero are NaN rather than a division by zero.

    Raises:
        NoBandsInRangeError: If either range holds no usable band.
    """
    visible = _average(cube, red)
    infrared = _average(cube, near_infrared)
    total = infrared + visible
    difference: NDArray[np.float64] = np.divide(
        infrared - visible,
        total,
        out=np.full(total.shape, np.nan),
        where=total != 0,
    )
    return difference


def rock(
    cube: Cube,
    *,
    vegetation_ndvi: float,
    dark_reflectance: float,
    red: tuple[float, float],
    near_infrared: tuple[float, float],
    shortwave: tuple[float, float],
) -> NDArray[np.bool_]:
    """Pixels worth matching against a mineral.

    Args:
        cube: Scene to mask.
        vegetation_ndvi: Pixels greener than this are dropped.
        dark_reflectance: Pixels darker than this in the shortwave are dropped.
        red: Wavelengths to average as red.
        near_infrared: Wavelengths to average as near infrared.
        shortwave: Wavelengths whose average stands for how bright a pixel is
            in the region the minerals are matched in.

    Returns:
        NDArray[np.bool_]: Mask of shape (row, column), true where the pixel
            holds usable measurements of exposed ground.

    Raises:
        NoBandsInRangeError: If any of the ranges holds no usable band.
    """
    usable = cube.valid_mask()
    green = greenness(cube, red, near_infrared)
    brightness = _average(cube, shortwave)

    bare = usable & ~(green > vegetation_ndvi)
    lit = bare & ~(brightness < dark_reflectance) & np.isfinite(brightness)

    total = usable.size
    logger.info(
        "{} of {} pixels hold a usable measurement ({:.1f}%)",
        int(usable.sum()),
        total,
        100 * usable.sum() / total,
    )
    logger.info(
        "greener than {}: {} pixels dropped ({:.1f}% of usable)",
        vegetation_ndvi,
        int(usable.sum() - bare.sum()),
        100 * (usable.sum() - bare.sum()) / max(int(usable.sum()), 1),
    )
    logger.info(
        "darker than {} in the shortwave: {} more dropped ({:.1f}% of usable)",
        dark_reflectance,
        int(bare.sum() - lit.sum()),
        100 * (bare.sum() - lit.sum()) / max(int(usable.sum()), 1),
    )
    logger.info(
        "{} pixels left to match against ({:.1f}% of the scene)",
        int(lit.sum()),
        100 * lit.sum() / total,
    )
    return lit


def _average(cube: Cube, wavelengths: tuple[float, float]) -> NDArray[np.float64]:
    """Mean reflectance across a range of wavelengths, per pixel.

    Averaged by hand rather than with `nanmean`, which warns on the empty
    slices that off-strip pixels produce.
    """
    low, high = wavelengths
    bands = cube.read_bands(cube.bands_between(low, high))
    measured = np.isfinite(bands)
    counted = measured.sum(axis=0)
    mean: NDArray[np.float64] = np.divide(
        np.where(measured, bands, 0.0).sum(axis=0),
        counted,
        out=np.full(counted.shape, np.nan),
        where=counted > 0,
    )
    return mean
