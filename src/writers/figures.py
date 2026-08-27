"""Drawing the results.

Four figures, each answering a question the numbers alone leave open: what the
map looks like, whether the two dates agree, whether the confidence layer means
anything, and whether a matched pixel really resembles the mineral it was given.

The colours are fixed per mineral and checked for colour-vision separation, so
the same mineral is the same colour in every figure and no two are told apart
by hue alone — every figure carries a legend or labels as well.
"""

from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.warp import transform

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ..agreement import Agreement
from ..spectra.continuum import remove
from ..spectra.mapping import Mapped

COLOURS = {
    "alunite": "#2a78d6",
    "kaolinite_group": "#eb6834",
    "muscovite": "#1baf7a",
    "carbonate": "#4a3aa7",
}
OTHER = "#8a8780"
UNNAMED = "#45423a"
SKIPPED = "#0b0a09"
VISIBLE = 50
INK = "#1b1915"
MUTED = "#6b6559"
DPI = 160


def mineral_map(mapped: Mapped, path: Path, title: str) -> Path:
    """Draw the map, with everything it declined to name left dark.

    Args:
        mapped: The map to draw.
        path: File to write.
        title: What the scene is.

    Returns:
        Path: The file written.
    """
    picture = np.full((*mapped.labels.shape, 3), _rgb(SKIPPED), dtype=np.uint8)
    picture[np.isfinite(mapped.depth)] = _rgb(UNNAMED)
    for index, group in enumerate(mapped.groups):
        picture[mapped.labels == index] = _rgb(COLOURS.get(group, OTHER))

    figure, axes = plt.subplots(
        figsize=(7.2, 7.2 * picture.shape[0] / picture.shape[1])
    )
    axes.imshow(picture, interpolation="nearest")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.spines[:].set_visible(False)
    axes.set_title(title, color=INK, fontsize=11, loc="left", pad=24)

    shown = [
        (group, COLOURS.get(group, OTHER))
        for index, group in enumerate(mapped.groups)
        if (mapped.labels == index).sum() >= VISIBLE
    ]
    axes.legend(
        handles=[_swatch(colour, group.replace("_", " ")) for group, colour in shown]
        + [
            _swatch(UNNAMED, "no mineral named"),
            _swatch(SKIPPED, "not examined"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        handletextpad=0.6,
        columnspacing=1.6,
    )
    _label_corners(axes, mapped)
    return _save(figure, path)


def _label_corners(axes: plt.Axes, mapped: Mapped) -> None:
    """Write where the map is, so it can be found on any other map.

    Corners rather than a graticule: the scene is north-up in a projected
    grid, so four numbers place it exactly and a grid of them would only
    cover the picture.
    """
    left, bottom, right, top = mapped.grid.bounds
    eastings, northings = transform(
        CRS.from_epsg(mapped.grid.epsg),
        "EPSG:4326",
        [left, right, left, right],
        [top, top, bottom, bottom],
    )
    rows, columns = mapped.labels.shape
    places = (
        (0, 0, "left", "bottom"),
        (columns, 0, "right", "bottom"),
        (0, rows, "left", "top"),
        (columns, rows, "right", "top"),
    )
    for (x, y, ha, va), lon, lat in zip(places, eastings, northings, strict=True):
        axes.annotate(
            f"{abs(lat):.3f}\u00b0{'N' if lat >= 0 else 'S'} "
            f"{abs(lon):.3f}\u00b0{'E' if lon >= 0 else 'W'}",
            (x, y),
            xytext=(0, 4 if va == "bottom" else -4),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.5,
            color=MUTED,
            annotation_clip=False,
        )


def _swatch(colour: str, label: str) -> plt.Line2D:
    """One entry in a legend."""
    return plt.Line2D(
        [], [], marker="s", linestyle="", markersize=8, color=colour, label=label
    )


def agreement_matrix(result: Agreement, path: Path, scenes: tuple[str, str]) -> Path:
    """Draw how the two maps compare, as counts and as row shares.

    Args:
        result: The comparison.
        path: File to write.
        scenes: Names of the two scenes, first down and second across.

    Returns:
        Path: The file written.
    """
    names = [name.replace("_", " ") for name in result.columns()]
    counts = np.array([row for _, row, _ in result.rows()], dtype=float)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    figure, axes = plt.subplots(figsize=(7.2, 5.6))
    axes.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)
    axes.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=9)
    axes.set_yticks(range(len(names)), names, fontsize=9)
    axes.set_xlabel(f"{scenes[1]} said", color=MUTED, fontsize=9)
    axes.set_ylabel(f"{scenes[0]} said", color=MUTED, fontsize=9)
    axes.set_title(
        f"{100 * result.rate:.1f}% agreed, kappa {result.kappa:.2f}, "
        f"{result.compared:,} pixels",
        color=INK,
        fontsize=11,
        loc="left",
        pad=10,
    )

    for row in range(shares.shape[0]):
        for column in range(shares.shape[1]):
            axes.text(
                column,
                row,
                f"{counts[row, column]:,.0f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if shares[row, column] > 0.5 else INK,
            )
    return _save(figure, path)


def confidence(split: dict[str, tuple[float, int]], path: Path) -> Path:
    """Draw whether the pixels the maps called settled actually agreed more.

    Args:
        split: Agreement rate and pixel count, by confidence.
        path: File to write.

    Returns:
        Path: The file written.
    """
    labels = list(split)
    rates = [100 * split[label][0] for label in labels]
    counts = [split[label][1] for label in labels]

    figure, axes = plt.subplots(figsize=(6.4, 3.2))
    bars = axes.barh(labels, rates, height=0.5, color=[COLOURS["muscovite"], OTHER])
    axes.set_xlim(0, 100)
    axes.set_xlabel("pixels the two dates agreed on (%)", color=MUTED, fontsize=9)
    axes.set_title(
        "Does the confidence mean anything?", color=INK, fontsize=11, loc="left", pad=10
    )
    axes.spines[["top", "right", "left"]].set_visible(False)
    axes.tick_params(length=0, labelsize=9.5)

    for bar, rate, count in zip(bars, rates, counts, strict=True):
        axes.text(
            rate + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%   {count:,} px",
            va="center",
            fontsize=9,
            color=INK,
        )
    return _save(figure, path)


def spectrum_against_reference(
    wavelengths: NDArray[np.float64],
    pixel: NDArray[np.float64],
    reference: NDArray[np.float64],
    path: Path,
    title: str,
    *,
    reference_label: str,
    caption: str,
) -> Path:
    """Draw a pixel beside the mineral it was matched to, before and after.

    The left panel is what the sensor recorded, where brightness dominates and
    the two look nothing alike. The right is what the matcher compares, once
    the continuum is divided out and only shape remains.

    Args:
        wavelengths: Wavelength of each band, in nanometres.
        pixel: The pixel's reflectance.
        reference: The matched mineral's reflectance, on the same bands.
        path: File to write.
        title: What the pixel was matched to.
        reference_label: Which laboratory sample the mineral line is.
        caption: How this pixel was picked out of its group. A spectrum drawn
            without saying that invites the reader to assume it is typical
            when it may be the best one in the scene.

    Returns:
        Path: The file written.
    """
    figure, (raw, shaped) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    raw.plot(wavelengths, pixel, color=COLOURS["alunite"], linewidth=1.6, label="pixel")
    raw.plot(wavelengths, reference, color=MUTED, linewidth=1.6, label=reference_label)
    raw.set_title("as recorded", color=INK, fontsize=10, loc="left")
    raw.set_ylabel("reflectance", color=MUTED, fontsize=9)

    for values, colour, name in (
        (pixel, COLOURS["alunite"], "pixel"),
        (reference, MUTED, reference_label),
    ):
        shaped.plot(
            wavelengths,
            remove(wavelengths, values),
            color=colour,
            linewidth=1.6,
            label=name,
        )
    shaped.set_title("continuum removed", color=INK, fontsize=10, loc="left")
    shaped.set_ylabel("fraction of continuum", color=MUTED, fontsize=9)

    for axis in (raw, shaped):
        axis.set_xlabel("wavelength (nm)", color=MUTED, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8.5)
    raw.legend(frameon=False, fontsize=9, loc="lower left")
    figure.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left", y=1.04)
    figure.text(0.02, -0.06, caption, color=MUTED, fontsize=8.5, ha="left")
    return _save(figure, path)


def _rgb(colour: str) -> tuple[int, int, int]:
    """A hex colour as bytes."""
    return tuple(int(colour[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


def _save(figure: plt.Figure, path: Path) -> Path:
    """Write a figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path
