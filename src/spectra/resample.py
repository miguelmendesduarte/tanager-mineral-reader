"""Putting laboratory spectra onto the bands of a scene.

A library spectrum is recorded at far finer detail than a satellite records,
so it has to be averaged down onto the sensor's bands before the two can be
compared. Each band sees a range of wavelengths rather than a single one, with
a response that falls off either side of its centre, so the average is weighted
by that response: a Gaussian of the band's own full width at half maximum.

Averaging down is honest, inventing detail is not. A library coarser than the
sensor cannot be sharpened, and one sampled more thinly than the sensor's bands
would be interpolated rather than averaged, which the guard below refuses.
"""

import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import LibraryTooCoarseError
from ..readers.wavelengths import Wavelengths
from .splib07 import Spectrum

FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
RESPONSE_CUTOFF = 3.0


def convolve(spectrum: Spectrum, target: Wavelengths) -> NDArray[np.float64]:
    """Average a library spectrum onto a sensor's bands.

    Args:
        spectrum: Library spectrum, on its own instrument's grid.
        target: Bands to average onto, with their centres and widths.

    Returns:
        NDArray[np.float64]: One reflectance per target band. Bands the library
            does not cover are NaN rather than an extrapolation.

    Raises:
        LibraryTooCoarseError: If the library is sampled more thinly than the
            target bands are wide, which would make this an interpolation.
    """
    _reject_thin_sampling(spectrum, target)

    sigma = target.widths / FWHM_TO_SIGMA
    offsets = spectrum.wavelengths[None, :] - target.centres[:, None]
    weights = np.exp(-0.5 * (offsets / sigma[:, None]) ** 2)

    weights[np.abs(offsets) > RESPONSE_CUTOFF * sigma[:, None]] = 0.0
    weights *= np.isfinite(spectrum.reflectance)[None, :]

    total = weights.sum(axis=1)
    covered = total > 0
    values = np.full(target.centres.size, np.nan)
    values[covered] = (weights[covered] @ np.nan_to_num(spectrum.reflectance)) / total[
        covered
    ]

    return values


def _reject_thin_sampling(spectrum: Spectrum, target: Wavelengths) -> None:
    """Refuse a library whose channels are spaced wider than a target band.

    Averaging needs several library channels under each band. One channel or
    fewer is interpolation wearing a convolution's clothes, and the library
    ships copies pre-convolved to other sensors that would hit this.

    The library grids are evenly spaced, so their median spacing describes the
    whole grid and there is no window to choose.
    """
    if spectrum.wavelengths.size < 2:
        raise LibraryTooCoarseError(spectrum.name, np.inf, float(target.widths.min()))

    spacing = float(np.median(np.diff(spectrum.wavelengths)))
    narrowest = float(target.widths.min())
    if spacing > narrowest:
        raise LibraryTooCoarseError(spectrum.name, spacing, narrowest)


def blunter_bands(
    spectrum: Spectrum,
    target: Wavelengths,
    low: float,
    high: float,
) -> tuple[int, int]:
    """Count the bands a library spectrum is too blunt to model sharply.

    Not an error: those bands are usable, but come out smoother than a real
    measurement of the mineral would be, which shallows absorption depths
    without moving them.

    Both instruments vary in width across their range, so the comparison is
    band by band at matching wavelengths, and over a stated range. Comparing
    the widest channel of one against the narrowest band of the other reports
    a mismatch between two unrelated parts of the spectrum, and a mismatch
    outside the range being matched says nothing about the result.

    Args:
        spectrum: Library spectrum, on its own instrument's grid.
        target: Bands the spectrum is averaged onto.
        low: Lower bound of the range that matters, in nanometres.
        high: Upper bound of the range that matters, in nanometres.

    Returns:
        tuple[int, int]: How many bands in the range the library is blunter
            than, and how many bands the range holds.
    """
    inside = (target.centres >= low) & (target.centres <= high)
    library_widths = np.interp(target.centres, spectrum.wavelengths, spectrum.widths)
    return int((inside & (library_widths > target.widths)).sum()), int(inside.sum())
