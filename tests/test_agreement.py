"""Tests for comparing two maps of the same ground."""

import numpy as np
import pytest

from src.agreement import by_confidence, compare
from src.core.exceptions import NoOverlapError, NotIndependentError
from src.readers.grid import Grid
from src.spectra.mapping import UNCLASSIFIED, Mapped
from src.spectra.rejection import NoiseFloor, Resolution

GROUPS = ("alunite", "kaolinite_group", "muscovite")
PIXEL = 30.0
ORIGIN = (472020.0, 4172340.0)
STRIPS = ("first_strip", "second_strip")


def _map(
    labels: list[list[int]],
    *,
    upper_left: tuple[float, float] = ORIGIN,
    resolved: list[list[bool]] | None = None,
) -> Mapped:
    """A map of a given shape, placed on the shared grid."""
    grid = np.asarray(labels, dtype=np.intp)
    rows, columns = grid.shape
    settled = (
        np.asarray(resolved, dtype=np.bool_)
        if resolved is not None
        else np.ones(grid.shape, dtype=np.bool_)
    )
    return Mapped(
        labels=grid,
        groups=GROUPS,
        angle=np.zeros(grid.shape),
        depth=np.zeros(grid.shape),
        margin=np.zeros(grid.shape),
        second=np.full(grid.shape, UNCLASSIFIED),
        resolved=settled,
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
            upper_left=upper_left,
            lower_right=(
                upper_left[0] + columns * PIXEL,
                upper_left[1] - rows * PIXEL,
            ),
            shape=(rows, columns),
        ),
    )


def test_two_identical_maps_agree_completely() -> None:
    labels = [[0, 1], [2, 0]]

    result = compare(_map(labels), _map(labels), strips=STRIPS)

    assert result.rate == pytest.approx(1.0)
    assert result.kappa == pytest.approx(1.0)
    assert result.compared == 4


def test_disagreement_is_counted_off_the_diagonal() -> None:
    result = compare(_map([[0, 0]]), _map([[0, 1]]), strips=STRIPS)

    assert result.rate == pytest.approx(0.5)
    assert result.matrix[0, 0] == 1
    assert result.matrix[0, 1] == 1


def test_unnamed_pixels_are_not_compared() -> None:
    """Neither map claimed anything there, so there is nothing to agree on."""
    result = compare(
        _map([[0, UNCLASSIFIED, 1]]),
        _map([[0, 2, UNCLASSIFIED]]),
        strips=STRIPS,
    )

    assert result.compared == 1
    assert result.overlapping == 3


def test_kappa_discounts_agreement_reached_by_chance() -> None:
    """Two maps that call almost everything one mineral are not in accord."""
    mostly = [[0] * 9 + [1]]
    result = compare(_map(mostly), _map([[0] * 8 + [1, 0]]), strips=STRIPS)

    assert result.rate == pytest.approx(0.8)
    assert result.kappa < result.rate


def test_each_group_gets_its_own_agreement() -> None:
    result = compare(_map([[0, 0, 1]]), _map([[0, 0, 2]]), strips=STRIPS)
    shares = result.per_group()

    assert shares["alunite"] == pytest.approx(1.0)
    assert shares["kaolinite_group"] == pytest.approx(0.0)


def test_two_pieces_of_one_pass_are_refused() -> None:
    """They are one observation, and would agree with themselves."""
    with pytest.raises(NotIndependentError):
        compare(_map([[0]]), _map([[0]]), strips=("same_strip", "same_strip"))


def test_maps_of_different_ground_are_refused() -> None:
    far = (ORIGIN[0] + 100 * PIXEL, ORIGIN[1])

    with pytest.raises(NoOverlapError):
        compare(_map([[0]]), _map([[0]], upper_left=far), strips=STRIPS)


def test_only_the_shared_ground_is_compared() -> None:
    """The second map starts one pixel right of the first."""
    shifted = (ORIGIN[0] + PIXEL, ORIGIN[1])

    result = compare(
        _map([[0, 1, 2]]),
        _map([[1, 2]], upper_left=shifted),
        strips=STRIPS,
    )

    assert result.compared == 2
    assert result.rate == pytest.approx(1.0)


def test_confidence_splits_the_agreement_in_two() -> None:
    """The test the confidence layer has to pass, in miniature."""
    first = _map([[0, 0, 0, 0]], resolved=[[True, True, False, False]])
    second = _map([[0, 0, 1, 1]], resolved=[[True, True, False, False]])

    split = by_confidence(first, second)

    assert split["both settled"] == (pytest.approx(1.0), 2)
    assert split["either unsettled"] == (pytest.approx(0.0), 2)


def test_the_matrix_reads_out_as_named_rows() -> None:
    result = compare(_map([[0, 0, 1]]), _map([[0, 1, 1]]), strips=STRIPS)

    assert result.columns() == ["alunite", "kaolinite_group"]
    assert result.rows() == [
        ("alunite", [1, 1], 2),
        ("kaolinite_group", [0, 1], 1),
    ]


def test_groups_neither_map_used_are_left_out_of_the_matrix() -> None:
    """A row of zeros for a mineral nobody found is noise in a table."""
    result = compare(_map([[0]]), _map([[0]]), strips=STRIPS)

    assert result.columns() == ["alunite"]
    assert [name for name, _, _ in result.rows()] == ["alunite"]
