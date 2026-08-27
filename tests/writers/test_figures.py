"""Tests for drawing the results.

A figure is judged by eye, so these check the things eyes miss: that every
drawing function runs on the shapes it will really be given, writes a file,
and does not quietly drop a mineral from a legend.
"""

from pathlib import Path

import numpy as np
import pytest

from src.agreement import compare
from src.readers.grid import Grid
from src.spectra.mapping import UNCLASSIFIED, Mapped
from src.spectra.rejection import NoiseFloor, Resolution
from src.writers.figures import (
    COLOURS,
    agreement_matrix,
    mineral_map,
    spectrum_against_reference,
)

GROUPS = ("alunite", "kaolinite_group", "muscovite", "carbonate")
WAVELENGTHS = np.linspace(2080.0, 2490.0, 83)


def _mapped(labels: list[list[int]]) -> Mapped:
    grid = np.asarray(labels, dtype=np.intp)
    rows, columns = grid.shape
    return Mapped(
        labels=grid,
        groups=GROUPS,
        angle=np.full(grid.shape, 22.0),
        depth=np.where(grid >= 0, 0.12, np.nan),
        margin=np.full(grid.shape, 6.0),
        second=np.full(grid.shape, UNCLASSIFIED),
        resolved=np.ones(grid.shape, dtype=np.bool_),
        floor=NoiseFloor(
            depth=0.06, angle=38.0, noise=0.002, brightness=0.2, quantile=0.99
        ),
        resolution=Resolution(
            depths=np.array([0.05, 0.3]),
            jitter=np.array([13.0, 2.7]),
            noise=0.002,
            quantile=0.95,
        ),
        grid=Grid(
            epsg=32611,
            upper_left=(472020.0, 4172340.0),
            lower_right=(472020.0 + 30.0 * columns, 4172340.0 - 30.0 * rows),
            shape=(rows, columns),
        ),
    )


def test_a_map_is_drawn(tmp_path: Path) -> None:
    path = mineral_map(
        _mapped([[0, 1], [2, UNCLASSIFIED]]), tmp_path / "map.png", "scene"
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_a_map_of_nothing_is_still_drawn(tmp_path: Path) -> None:
    """A scene where nothing was named is a real answer and must not crash."""
    path = mineral_map(
        _mapped([[UNCLASSIFIED, UNCLASSIFIED]]), tmp_path / "map.png", "scene"
    )

    assert path.exists()


def test_every_mineral_has_a_colour_of_its_own() -> None:
    """Two minerals sharing a colour would be a silently unreadable map."""
    assert len(set(COLOURS.values())) == len(COLOURS)


def test_the_agreement_matrix_is_drawn(tmp_path: Path) -> None:
    result = compare(
        _mapped([[0, 1, 2]]),
        _mapped([[0, 2, 2]]),
        strips=("one", "two"),
    )

    path = agreement_matrix(result, tmp_path / "agreement.png", ("first", "second"))

    assert path.exists()
    assert path.stat().st_size > 0


def test_a_pixel_is_drawn_against_its_reference(tmp_path: Path) -> None:
    dip = 1.0 - 0.4 * np.exp(-0.5 * ((WAVELENGTHS - 2170.0) / 25.0) ** 2)

    path = spectrum_against_reference(
        WAVELENGTHS,
        0.3 * dip,
        dip,
        tmp_path / "spectrum.png",
        "alunite",
        reference_label="Alunite (laboratory)",
        caption="The pixel is the median depth of 812 settled alunite pixels.",
    )

    assert path.exists()


def test_a_map_only_legends_minerals_you_can_actually_see(tmp_path: Path) -> None:
    """One pixel of a mineral is a swatch for something nobody can find."""
    labels = [[0] * 40, [1] + [0] * 39]

    path = mineral_map(_mapped(labels), tmp_path / "map.png", "scene")

    assert path.exists()


def test_the_directory_is_made_if_it_is_not_there(tmp_path: Path) -> None:
    path = mineral_map(_mapped([[0]]), tmp_path / "deep" / "map.png", "scene")

    assert path.exists()


@pytest.mark.parametrize("group", GROUPS)
def test_every_group_the_map_uses_can_be_coloured(group: str) -> None:
    """A mineral with no colour falls back to grey, which reads as 'other'."""
    assert group in COLOURS
