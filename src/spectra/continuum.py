"""Taking the shape of a spectrum away from its brightness.

A reflectance spectrum is dominated by how bright the surface is and how that
brightness drifts across wavelength. Two samples of the same mineral, one in
sun and one in shade, look nothing alike until that envelope is divided out;
two different minerals of the same brightness look alike until it is.

The envelope is the upper convex hull: the taut line laid over the top of the
spectrum, touching its peaks and bridging its absorptions. Dividing by it
leaves 1.0 where the spectrum touches the hull and dips towards 0 inside every
absorption, so what remains is shape alone.
"""

import numpy as np
from numpy.typing import NDArray


def remove(
    wavelengths: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Divide a spectrum by its upper convex hull.

    Args:
        wavelengths: Wavelength of each point, increasing.
        values: Reflectance of each point. Points that are not finite take no
            part in the hull and come back as NaN.

    Returns:
        NDArray[np.float64]: The spectrum with its envelope divided out, 1.0 on
            the hull and lower inside absorptions.
    """
    removed = np.full(values.shape, np.nan)
    usable = np.isfinite(values) & np.isfinite(wavelengths)
    if usable.sum() < 2:
        return removed

    hull = _upper_hull(wavelengths[usable], values[usable])
    removed[usable] = np.divide(
        values[usable],
        hull,
        out=np.full(hull.shape, np.nan),
        where=hull != 0,
    )
    return removed


def deepest_feature(
    wavelengths: NDArray[np.float64],
    values: NDArray[np.float64],
    low: float,
    high: float,
) -> tuple[float, float]:
    """Find the strongest absorption inside a range of wavelengths.

    The continuum is fitted across the range asked for, not the whole spectrum,
    so a steep roll-off outside it cannot masquerade as the deepest feature.

    Args:
        wavelengths: Wavelength of each point, increasing.
        values: Reflectance of each point.
        low: Lower bound of the range, in nanometres.
        high: Upper bound of the range, in nanometres.

    Returns:
        tuple[float, float]: Wavelength of the deepest absorption and how deep
            it is, as a fraction of the continuum. Both NaN if the range holds
            nothing usable.
    """
    inside = (wavelengths >= low) & (wavelengths <= high) & np.isfinite(values)
    if inside.sum() < 2:
        return float("nan"), float("nan")

    window, spectrum = wavelengths[inside], values[inside]
    removed = remove(window, spectrum)
    deepest = int(np.nanargmin(removed))
    return float(window[deepest]), float(1.0 - removed[deepest])


def _upper_hull(
    wavelengths: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The taut line laid over the top of a spectrum, sampled at every point.

    Walks left to right keeping the vertices that still turn downwards, which
    is Andrew's monotone chain over points already sorted by wavelength.

    The walk runs over plain Python floats rather than the arrays themselves.
    Reading a single element out of a numpy array builds a scalar object, which
    costs several times the arithmetic done with it, and this loop reads far
    more often than it computes.
    """
    lengths = wavelengths.tolist()
    heights = values.tolist()

    vertices: list[int] = [0]
    for candidate in range(1, len(lengths)):
        while len(vertices) >= 2:
            last, previous = vertices[-1], vertices[-2]
            rises = (heights[last] - heights[previous]) * (
                lengths[candidate] - lengths[previous]
            )
            spans = (heights[candidate] - heights[previous]) * (
                lengths[last] - lengths[previous]
            )
            if rises > spans:
                break
            vertices.pop()
        vertices.append(candidate)

    return np.interp(wavelengths, wavelengths[vertices], values[vertices])
