"""Tests for the map grid."""

import pytest

from src.core.exceptions import (
    DifferentPixelSizesError,
    DifferentProjectionsError,
    GridMetadataError,
    OffLatticeError,
)
from src.readers.grid import Grid

from .conftest import EPSG, LOWER_RIGHT, STRUCT_METADATA, UPPER_LEFT


def _grid() -> Grid:
    return Grid.from_metadata(STRUCT_METADATA, epsg=EPSG, shape=(4, 5))


def test_grid_reads_the_corners_from_the_metadata() -> None:
    grid = _grid()

    assert grid.upper_left == UPPER_LEFT
    assert grid.lower_right == LOWER_RIGHT
    assert grid.epsg == EPSG


def test_grid_derives_the_pixel_size_from_the_corners() -> None:
    assert _grid().pixel_size == (30.0, 30.0)


def test_grid_builds_a_transform_that_walks_down_the_scene() -> None:
    left, width, row_rotation, top, column_rotation, height = _grid().transform

    assert (left, top) == UPPER_LEFT
    assert (width, height) == (30.0, -30.0)
    assert (row_rotation, column_rotation) == (0.0, 0.0)


def test_grid_reports_its_bounds() -> None:
    left, bottom, right, top = _grid().bounds

    assert (left, top) == UPPER_LEFT
    assert (right, bottom) == LOWER_RIGHT


def test_grid_reports_metadata_without_corners() -> None:
    with pytest.raises(GridMetadataError, match="UpperLeftPointMtrs"):
        Grid.from_metadata("GROUP=GridStructure", epsg=EPSG, shape=(4, 5))


def _placed_grid(
    epsg: int = 32611,
    upper_left: tuple[float, float] = (472020.0, 4172340.0),
    shape: tuple[int, int] = (100, 100),
    pixel: float = 30.0,
) -> Grid:
    """A grid of a given size, placed on the map."""
    left, top = upper_left
    rows, columns = shape
    return Grid(
        epsg=epsg,
        upper_left=upper_left,
        lower_right=(left + columns * pixel, top - rows * pixel),
        shape=shape,
    )


def test_offset_to_locates_a_grid_inside_another() -> None:
    outer = _placed_grid()
    inner = _placed_grid(upper_left=(472020.0 + 109 * 30.0, 4172340.0 - 173 * 30.0))

    assert outer.offset_to(inner) == (173, 109)


def test_offset_to_is_zero_for_the_same_placed_grid() -> None:
    grid = _placed_grid()

    assert grid.offset_to(grid) == (0, 0)


def test_offset_to_can_be_negative() -> None:
    """A scene may begin above and to the left of the one it is located in."""
    outer = _placed_grid()
    inner = _placed_grid(upper_left=(472020.0 - 60.0, 4172340.0 + 90.0))

    assert outer.offset_to(inner) == (-3, -2)


def test_offset_to_refuses_a_different_projection() -> None:
    with pytest.raises(DifferentProjectionsError):
        _placed_grid().offset_to(_placed_grid(epsg=32612))


def test_offset_to_refuses_a_different_pixel_size() -> None:
    with pytest.raises(DifferentPixelSizesError):
        _placed_grid().offset_to(_placed_grid(pixel=20.0))


def test_offset_to_refuses_a_half_pixel_shift() -> None:
    """Resampling is not implemented, so an off-lattice pair is refused."""
    with pytest.raises(OffLatticeError):
        _placed_grid().offset_to(_placed_grid(upper_left=(472035.0, 4172340.0)))
