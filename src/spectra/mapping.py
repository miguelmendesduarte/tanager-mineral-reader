"""Turning a scene into a map of what is on the ground.

This is where the pieces meet: the pixels worth looking at, the references
averaged onto the scene's own bands, the match, and the refusal to name a pixel
that says no more than noise would have. What comes back is a label for every
pixel of the scene and the numbers behind each decision, so a map can be drawn
and then argued with.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..core.config import Settings
from ..readers.cube import Cube
from ..readers.masks import rock
from .matching import Match, match
from .rejection import NoiseFloor, Resolution, measure, measure_resolution
from .resample import convolve
from .splib07 import Spectrum, read_spectra

UNCLASSIFIED = -1
NOISE_SAMPLE = 7


@dataclass(frozen=True)
class Mapped:
    """What was found in a scene, laid back out over its pixels.

    Attributes:
        labels: Index into `groups` for every pixel of the scene, or
            UNCLASSIFIED where nothing could be said.
        groups: Names of the mineral groups, in label order.
        angle: Angle to the closest reference, per pixel; NaN where unmatched.
        depth: Deepest absorption, per pixel; NaN where unmatched.
        margin: Degrees by which the winner beat the runner-up.
        second: Group the runner-up belongs to, per pixel. Read it together
            with `resolved`: where a ranking is not settled, this is the other
            mineral the pixel could equally be.
        resolved: Whether the winner beat the runner-up by more than noise
            could account for.
        floor: The noise floor the decisions were made against.
        resolution: The ranking precision the decisions were made against.
    """

    labels: NDArray[np.intp]
    groups: tuple[str, ...]
    angle: NDArray[np.float64]
    depth: NDArray[np.float64]
    margin: NDArray[np.float64]
    second: NDArray[np.intp]
    resolved: NDArray[np.bool_]
    floor: NoiseFloor
    resolution: Resolution

    def pairs(self) -> dict[str, int]:
        """How many pixels each unsettled pair of minerals covers.

        A pixel on the boundary between two zones is a mixture of both, and
        the honest reading of it is the pair rather than whichever edged
        ahead. Keys are the two group names, in a fixed order so that the same
        pair always reads the same way.
        """
        unsettled = self.named & ~self.resolved & (self.second != UNCLASSIFIED)
        counts: dict[str, int] = {}
        for first, other in zip(
            self.labels[unsettled].tolist(),
            self.second[unsettled].tolist(),
            strict=True,
        ):
            key = " + ".join(sorted((self.groups[first], self.groups[other])))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    @property
    def named(self) -> NDArray[np.bool_]:
        """Pixels that were given a mineral."""
        given: NDArray[np.bool_] = self.labels != UNCLASSIFIED
        return given

    def share(self) -> dict[str, float]:
        """Fraction of the named pixels that went to each group."""
        named = int(self.named.sum())
        if not named:
            return {}
        counts = np.bincount(self.labels[self.named], minlength=len(self.groups))
        return {
            group: float(count / named)
            for group, count in zip(self.groups, counts, strict=True)
            if count
        }


def map_scene(path: Path, settings: Settings) -> Mapped:
    """Work out which mineral, if any, each pixel of a scene shows.

    Args:
        path: Surface reflectance product of the scene.
        settings: Which references to match against, and how to mask.

    Returns:
        Mapped: A label per pixel and the numbers behind it.
    """
    low, high = settings.match_range

    with Cube(path) as cube:
        bands = cube.wavelengths
        selected = cube.bands_between(low, high)
        window = bands.centres[selected]
        block = cube.read_bands(selected)
        usable = rock(
            cube,
            vegetation_ndvi=settings.vegetation_ndvi,
            dark_reflectance=settings.dark_reflectance,
            red=settings.red_nm,
            near_infrared=settings.near_infrared_nm,
            shortwave=(low, high),
        )
        noise = cube.noise(selected, step=NOISE_SAMPLE)
        spectra = read_spectra(settings.splib07_archive, settings.references)
        references = np.stack([
            convolve(spectrum, bands)[selected] for spectrum in spectra
        ])

    rows, columns = usable.shape
    flat = np.asarray(block.reshape(window.size, -1).T, dtype=np.float64)
    where = np.flatnonzero(usable.reshape(-1))
    where = where[np.isfinite(flat[where]).all(axis=1)]

    logger.info("Matching {} pixels against {} references", where.size, len(spectra))
    result = match(flat[where], references, window)

    floor = measure(
        references,
        window,
        noise=noise,
        brightness=float(np.nanmedian(flat[where])),
    )
    logger.info(
        "Noise alone fakes a depth of {:.4f} and comes within {:.1f} degrees, "
        "so a pixel has to beat both",
        floor.depth,
        floor.angle,
    )

    resolution = measure_resolution(references, window, noise=noise)
    return _lay_out(
        result, floor, resolution, spectra, settings, where, (rows, columns)
    )


def group_indices(settings: Settings, names: list[str]) -> NDArray[np.intp]:
    """Which group each reference reports as, as an index into the groups.

    References that are not minerals come back as UNCLASSIFIED. They are
    matched against like any other, and a pixel that lands on one is left
    unnamed rather than given the mineral it least resembles.

    Args:
        settings: Holds the groups and the non-mineral references.
        names: Reference spectra, in the order they were matched.

    Returns:
        NDArray[np.intp]: One index per reference.

    Raises:
        KeyError: If a reference is not configured at all.
    """
    groups = tuple(dict.fromkeys(settings.mineral_groups))
    return np.array(
        [
            groups.index(group) if (group := settings.group_of(name)) else UNCLASSIFIED
            for name in names
        ],
        dtype=np.intp,
    )


def _lay_out(
    result: Match,
    floor: NoiseFloor,
    resolution: Resolution,
    spectra: list[Spectrum],
    settings: Settings,
    where: NDArray[np.intp],
    shape: tuple[int, int],
) -> Mapped:
    """Put the matched pixels back where they came from in the scene."""
    groups = tuple(dict.fromkeys(settings.mineral_groups))
    per_reference = group_indices(settings, [spectrum.name for spectrum in spectra])

    accepted = floor.accepts(result.depth, result.best_angle)
    chosen = np.where(accepted, per_reference[result.best], UNCLASSIFIED)
    settled = resolution.resolves(result.depth, result.margin)

    pixels = shape[0] * shape[1]
    labels = np.full(pixels, UNCLASSIFIED, dtype=np.intp)
    second = np.full(pixels, UNCLASSIFIED, dtype=np.intp)
    resolved = np.zeros(pixels, dtype=np.bool_)
    angle = np.full(pixels, np.nan)
    depth = np.full(pixels, np.nan)
    margin = np.full(pixels, np.nan)

    labels[where] = chosen
    second[where] = np.where(accepted, per_reference[result.runner_up], UNCLASSIFIED)
    resolved[where] = settled
    angle[where] = result.best_angle
    depth[where] = result.depth
    margin[where] = result.margin

    mapped = Mapped(
        labels=labels.reshape(shape),
        groups=groups,
        angle=angle.reshape(shape),
        depth=depth.reshape(shape),
        margin=margin.reshape(shape),
        second=second.reshape(shape),
        resolved=resolved.reshape(shape),
        floor=floor,
        resolution=resolution,
    )
    _report(mapped, result, accepted, per_reference)
    return mapped


def _report(
    mapped: Mapped,
    result: Match,
    accepted: NDArray[np.bool_],
    per_reference: NDArray[np.intp],
) -> None:
    """Say what was kept, what was refused, and why."""
    matched = accepted.size
    not_mineral = int((accepted & (per_reference[result.best] == UNCLASSIFIED)).sum())
    named = int(mapped.named.sum())

    logger.info(
        "{} pixels named, {} refused as no better than noise, {} recognised as "
        "something that is not a mineral",
        named,
        matched - int(accepted.sum()),
        not_mineral,
    )
    firm = int((mapped.named & mapped.resolved).sum())
    logger.info(
        "{} of those are settled; the other {} sit between two minerals and "
        "could be either",
        firm,
        named - firm,
    )
    for group, share in sorted(mapped.share().items(), key=lambda item: -item[1]):
        selected = mapped.labels == mapped.groups.index(group)
        settled = int((selected & mapped.resolved).sum())
        logger.info(
            "   {:18s} {:5.1f}%  {:3.0f}% settled",
            group,
            100 * share,
            100 * settled / max(int(selected.sum()), 1),
        )
    for pair, count in list(mapped.pairs().items())[:4]:
        logger.info("   unsettled: {:34s} {}", pair, count)
