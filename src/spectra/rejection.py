"""Working out when a pixel matches nothing at all.

Picking the closest of several references always succeeds, so on its own it
says nothing: a flat patch of gravel is given a mineral name as readily as an
outcrop. Something has to be able to answer "none of these".

The tempting test is whether a pixel is as close to its reference as a correct
match would be. It cannot be measured. A correct match on a laboratory mineral
plus sensor noise lands within a few degrees, but no ground pixel is a
laboratory mineral: at 30 m it is a mixture of several, with dust and grain
size on top. Calibrating against pure minerals rejects everything, and
calibrating against assumed mixtures only moves the answer to wherever the
assumed mixing was set.

What can be measured is the opposite question: what could noise alone have
faked? Take a flat spectrum as bright as the scene, add noise of the kind the
sensor actually delivers, and match it. It comes back with an absorption depth
and an angle like any other pixel, and those are what meaning nothing looks
like. A pixel must beat both to be worth a name.

Nothing here is chosen except the quantile, which says how often noise is
allowed to pass for a mineral.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .matching import absorption, angles_between, match

TRIALS = 20000
QUANTILE = 0.99
DEPTHS = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
RESOLUTION_TRIALS = 300
RESOLUTION_QUANTILE = 0.95


@dataclass(frozen=True)
class NoiseFloor:
    """What a pixel of pure noise looks like, in this scene.

    Attributes:
        depth: Deepest absorption noise fabricates, at the chosen quantile.
        angle: Closest a noise pixel comes to any reference, at the same
            quantile from the other end.
        noise: Reflectance noise the floor was measured with.
        brightness: Reflectance the noise was measured against, which sets how
            deep a given amount of noise appears once the continuum is removed.
        quantile: Share of noise pixels the floor is set to exclude.
    """

    depth: float
    angle: float
    noise: float
    brightness: float
    quantile: float

    def accepts(
        self,
        depth: NDArray[np.float64],
        angle: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Whether each pixel says more than noise would have.

        Both tests are needed and they ask different things. Depth asks whether
        anything is absorbing at all; a pixel that fails it is flat, and its
        nearest reference is an accident. Angle asks whether what absorbs has
        the shape of something in the list; a pixel can have a deep, real
        absorption belonging to a mineral nobody put in the config, and only
        this test refuses it.

        Args:
            depth: Deepest absorption in each pixel.
            angle: Angle from each pixel to its closest reference, in degrees.

        Returns:
            NDArray[np.bool_]: True where the pixel is worth naming. A pixel
                with no measurable shape is never accepted.
        """
        measured = np.isfinite(depth) & np.isfinite(angle)
        return measured & (depth > self.depth) & (angle < self.angle)


def measure(
    references: NDArray[np.float64],
    wavelengths: NDArray[np.float64],
    *,
    noise: float,
    brightness: float,
    trials: int = TRIALS,
    quantile: float = QUANTILE,
    seed: int = 0,
) -> NoiseFloor:
    """Match pure noise against the references and see what it manages.

    Args:
        references: Reflectance of each reference, shaped (references, bands).
        wavelengths: Wavelength of each band, in nanometres.
        noise: Standard deviation of the sensor's reflectance noise.
        brightness: Typical reflectance of the scene over the same bands.
        trials: Noise pixels to match. Enough are needed that the tail is
            measured rather than estimated.
        quantile: Share of noise pixels to exclude. At 0.99 a hundredth of
            them would still pass for a mineral.
        seed: Fixes the noise, so the floor is the same on every run.

    Returns:
        NoiseFloor: The depth and angle a pixel has to beat.
    """
    generator = np.random.default_rng(seed)
    flat = brightness + generator.normal(0.0, noise, (trials, wavelengths.size))
    faked = match(flat, references, wavelengths)

    return NoiseFloor(
        depth=float(np.nanquantile(faked.depth, quantile)),
        angle=float(np.nanquantile(faked.best_angle, 1.0 - quantile)),
        noise=noise,
        brightness=brightness,
        quantile=quantile,
    )


@dataclass(frozen=True)
class Resolution:
    """How far noise alone can move a ranking, by absorption depth.

    Attributes:
        depths: Band depths the jitter was measured at.
        jitter: Degrees the margin between first and second place moves at
            each of those depths.
        noise: Reflectance noise the jitter was measured with.
        quantile: Share of noisy copies the jitter covers.
    """

    depths: NDArray[np.float64]
    jitter: NDArray[np.float64]
    noise: float
    quantile: float

    def at(self, depth: NDArray[np.float64]) -> NDArray[np.float64]:
        """The jitter at any depth, between and beyond the measured ones."""
        return np.interp(depth, self.depths, self.jitter)

    def resolves(
        self,
        depth: NDArray[np.float64],
        margin: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Whether first place beat second by more than noise could account for.

        Args:
            depth: Deepest absorption in each pixel.
            margin: Degrees by which the closest reference beat the next.

        Returns:
            NDArray[np.bool_]: True where the order of the two is settled. False
                does not mean the pixel is unmatched — it means the pixel sits
                between two references and which one wins is not decidable.
        """
        measured = np.isfinite(depth) & np.isfinite(margin)
        return measured & (margin > self.at(depth))


def measure_resolution(
    references: NDArray[np.float64],
    wavelengths: NDArray[np.float64],
    *,
    noise: float,
    depths: tuple[float, ...] = DEPTHS,
    trials: int = RESOLUTION_TRIALS,
    quantile: float = RESOLUTION_QUANTILE,
    seed: int = 0,
) -> Resolution:
    """Measure how much noise alone shifts the gap between first and second.

    A reference is diluted to a given depth, noise is added, and the gap
    between its own reference and the next nearest is compared with the gap it
    would have had with no noise at all. The spread of that difference is how
    precisely any ranking can be known at that depth.

    This is the pure-mineral calibration that fails as a test of whether a
    pixel matches at all — real ground is never a laboratory mineral. It is
    the right test for whether two references can be told apart, because that
    question is about the precision of a measurement rather than the realism
    of a model.

    Args:
        references: Reflectance of each reference, shaped (references, bands).
        wavelengths: Wavelength of each band, in nanometres.
        noise: Standard deviation of the sensor's reflectance noise.
        depths: Band depths to measure at.
        trials: Noisy copies of each reference at each depth.
        quantile: Share of the movement to cover.
        seed: Fixes the noise, so the result is the same on every run.

    Returns:
        Resolution: The jitter, depth by depth.
    """
    generator = np.random.default_rng(seed)
    truth = absorption(references, wavelengths)
    count = references.shape[0]

    clean = angles_between(truth, truth)
    np.fill_diagonal(clean, np.inf)
    settled = np.nanmin(clean, axis=1)

    moved = []
    for depth in depths:
        peak = np.nanmax(truth, axis=1, keepdims=True)
        scale = np.divide(depth, peak, out=np.zeros_like(peak), where=peak > 0)
        noisy = (1.0 - truth * scale)[:, None, :] + generator.normal(
            0.0, noise, (count, trials, wavelengths.size)
        )
        shapes = absorption(noisy.reshape(-1, wavelengths.size), wavelengths)
        against = angles_between(shapes, truth)

        origin = np.repeat(np.arange(count), trials)
        rows = np.arange(origin.size)
        own = against[rows, origin]
        against[rows, origin] = np.inf
        gap = np.nanmin(against, axis=1) - own
        moved.append(float(np.nanquantile(np.abs(gap - settled[origin]), quantile)))

    return Resolution(
        depths=np.asarray(depths, dtype=np.float64),
        jitter=np.asarray(moved, dtype=np.float64),
        noise=noise,
        quantile=quantile,
    )
