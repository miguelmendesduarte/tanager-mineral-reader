"""Tests for the map grid."""

import pytest

from src.core.exceptions import GridMetadataError
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
