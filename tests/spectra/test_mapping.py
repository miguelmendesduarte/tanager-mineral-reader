"""Tests for laying a match back out over a scene."""

import numpy as np
import pytest

from src.core.config import Settings
from src.readers.grid import Grid
from src.spectra.mapping import UNCLASSIFIED, Mapped, group_indices
from src.spectra.rejection import NoiseFloor, Resolution

GROUPS = {"alunite": ("A1",), "kaolinite_group": ("K1", "K2"), "carbonate": ("C1",)}
NOT_MINERALS = ("Grass1", "Grass2")


def _settings() -> Settings:
    return Settings(mineral_groups=GROUPS, nonmineral_spectra=NOT_MINERALS)


def _mapped(
    labels: list[list[int]],
    second: list[list[int]] | None = None,
    resolved: list[list[bool]] | None = None,
) -> Mapped:
    grid = np.asarray(labels, dtype=np.intp)
    runner = (
        np.asarray(second, dtype=np.intp)
        if second
        else np.full(grid.shape, UNCLASSIFIED)
    )
    settled = (
        np.asarray(resolved, dtype=np.bool_)
        if resolved
        else np.ones(grid.shape, np.bool_)
    )
    return Mapped(
        labels=grid,
        groups=tuple(GROUPS),
        angle=np.zeros(grid.shape),
        depth=np.zeros(grid.shape),
        margin=np.zeros(grid.shape),
        second=runner,
        resolved=settled,
        floor=NoiseFloor(
            depth=0.06, angle=38.0, noise=0.002, brightness=0.2, quantile=0.99
        ),
        grid=Grid(
            epsg=32611,
            upper_left=(472020.0, 4172340.0),
            lower_right=(
                472020.0 + 30.0 * grid.shape[1],
                4172340.0 - 30.0 * grid.shape[0],
            ),
            shape=grid.shape,
        ),
        resolution=Resolution(
            depths=np.array([0.05, 0.3]),
            jitter=np.array([13.0, 2.7]),
            noise=0.002,
            quantile=0.95,
        ),
    )


def test_every_species_of_a_group_reports_as_that_group() -> None:
    """Two kaolin minerals are one class, so they share an index."""
    indices = group_indices(_settings(), ["A1", "K1", "K2", "C1"])

    assert indices.tolist() == [0, 1, 1, 2]


def test_a_reference_that_is_not_a_mineral_names_nothing() -> None:
    """It is matched against, but a pixel landing on it is left unnamed."""
    indices = group_indices(_settings(), ["A1", "Grass1", "Grass2"])

    assert indices.tolist() == [0, UNCLASSIFIED, UNCLASSIFIED]


def test_a_reference_nobody_configured_is_an_error() -> None:
    with pytest.raises(KeyError):
        group_indices(_settings(), ["Something_Else"])


def test_named_pixels_are_the_ones_with_a_group() -> None:
    mapped = _mapped([[0, UNCLASSIFIED], [2, 1]])

    assert mapped.named.tolist() == [[True, False], [True, True]]


def test_share_counts_only_the_named_pixels() -> None:
    mapped = _mapped([[0, 0], [1, UNCLASSIFIED]])

    assert mapped.share() == {
        "alunite": pytest.approx(2 / 3),
        "kaolinite_group": pytest.approx(1 / 3),
    }


def test_share_of_a_scene_where_nothing_was_named() -> None:
    """A blank map is a real answer, not a crash."""
    assert _mapped([[UNCLASSIFIED, UNCLASSIFIED]]).share() == {}


def test_an_unsettled_pixel_is_reported_as_the_pair_it_sits_between() -> None:
    """On a zone boundary a pixel is both minerals, not whichever edged ahead."""
    mapped = _mapped(
        [[0, 1], [2, 0]],
        second=[[1, 0], [0, 1]],
        resolved=[[False, False], [True, False]],
    )

    assert mapped.pairs() == {"alunite + kaolinite_group": 3}


def test_a_settled_pixel_is_not_reported_as_a_pair() -> None:
    mapped = _mapped([[0]], second=[[1]], resolved=[[True]])

    assert mapped.pairs() == {}


def test_a_tie_between_two_species_of_one_group_settles_that_group() -> None:
    """Kaolinite against dickite is one answer, however close the call.

    Both report as the kaolinite group, and the map never claims to tell the
    species apart, so the margin between them decides nothing.
    """
    mapped = _mapped([[1]], second=[[1]], resolved=[[True]])

    assert mapped.pairs() == {}
