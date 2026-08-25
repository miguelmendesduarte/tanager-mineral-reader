"""Deciding which mineral a pixel looks most like.

Every pixel and every reference has its continuum removed over the same bands,
which leaves the shape of their absorptions and nothing of their brightness.
The two are then compared by the angle between them: treat each spectrum's
absorption depths as a vector and measure how far apart the vectors point.
Angle ignores length, so a pixel where a mineral covers a third of the ground
scores the same as one where it covers all of it — only the shape decides.

Two things come back besides the winner. The runner-up says how close the
decision was, which matters where minerals are genuinely alike. The band depth
says how much there was to decide on at all, since a flat spectrum has a
nearest neighbour like any other, and it is meaningless.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import TooFewReferencesError
from .continuum import remove

MINIMUM_REFERENCES = 2


@dataclass(frozen=True)
class Match:
    """What each pixel was matched to, and how well.

    Attributes:
        best: Index of the closest reference, per pixel.
        best_angle: Angle to it in degrees; smaller is a closer match.
        runner_up: Index of the second closest reference.
        runner_up_angle: Angle to that one, in degrees.
        depth: Deepest absorption in the pixel, as a fraction of its continuum.
    """

    best: NDArray[np.intp]
    best_angle: NDArray[np.float64]
    runner_up: NDArray[np.intp]
    runner_up_angle: NDArray[np.float64]
    depth: NDArray[np.float64]

    def __len__(self) -> int:
        """Number of pixels matched."""
        return int(self.best.size)

    @property
    def margin(self) -> NDArray[np.float64]:
        """How much closer the winner was than the runner-up, in degrees.

        A small margin means the two references were nearly as good as each
        other, which is what ambiguity looks like when it is honest.
        """
        return self.runner_up_angle - self.best_angle


def match(
    spectra: NDArray[np.float64],
    references: NDArray[np.float64],
    wavelengths: NDArray[np.float64],
) -> Match:
    """Match each spectrum against every reference.

    The continuum is removed from both sides here rather than by the caller,
    so that it is removed the same way from each. Comparing a pixel that has
    been detrended against a reference that has not would measure the
    detrending.

    Args:
        spectra: Reflectance of each pixel, shaped (pixels, bands).
        references: Reflectance of each reference, shaped (references, bands),
            on the same bands as the spectra.
        wavelengths: Wavelength of each band, in nanometres.

    Returns:
        Match: The closest and second closest reference for every pixel.

    Raises:
        TooFewReferencesError: If fewer than two references are given, since
            without a runner-up there is no way to say how close the call was.
    """
    if references.shape[0] < MINIMUM_REFERENCES:
        raise TooFewReferencesError(references.shape[0])

    pixel_shape = _absorption(spectra, wavelengths)
    reference_shape = _absorption(references, wavelengths)

    angles = _angles(pixel_shape, reference_shape)
    ranked = np.argsort(angles, axis=1)
    best, runner_up = ranked[:, 0], ranked[:, 1]
    rows = np.arange(angles.shape[0])

    return Match(
        best=best,
        best_angle=angles[rows, best],
        runner_up=runner_up,
        runner_up_angle=angles[rows, runner_up],
        depth=np.nanmax(
            pixel_shape, axis=1, initial=0.0, where=np.isfinite(pixel_shape)
        ),
    )


def _absorption(
    spectra: NDArray[np.float64],
    wavelengths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """How far below its continuum each spectrum sits, band by band.

    Zero where the spectrum touches its continuum, larger inside absorptions,
    so a featureless spectrum comes back as zeros and has no direction at all.
    """
    removed = np.stack([remove(wavelengths, spectrum) for spectrum in spectra])
    return np.where(np.isfinite(removed), 1.0 - removed, 0.0)


def _angles(
    pixels: NDArray[np.float64],
    references: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Angle in degrees between every pixel and every reference."""
    pixel_lengths = np.linalg.norm(pixels, axis=1, keepdims=True)
    reference_lengths = np.linalg.norm(references, axis=1, keepdims=True)

    scale = pixel_lengths * reference_lengths.T
    cosines = np.divide(
        pixels @ references.T,
        scale,
        out=np.full(scale.shape, np.nan),
        where=scale > 0,
    )
    angles: NDArray[np.float64] = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return angles
