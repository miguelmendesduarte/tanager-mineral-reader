"""Tests for working out when a pixel matches nothing."""

import numpy as np
import pytest

from src.spectra.matching import match
from src.spectra.rejection import NoiseFloor, measure

WAVELENGTHS = np.linspace(2080.0, 2490.0, 83)
NOISE = 0.0022
BRIGHTNESS = 0.21


def _dip(centre: float, depth: float, width: float = 25.0) -> np.ndarray:
    """A spectrum with one absorption in it, on a flat continuum."""
    return 1.0 - depth * np.exp(-0.5 * ((WAVELENGTHS - centre) / width) ** 2)


def _references() -> np.ndarray:
    """Three references, absorbing at 2170, 2210 and 2340 nm."""
    return np.stack([_dip(2170.0, 0.5), _dip(2210.0, 0.4), _dip(2340.0, 0.3)])


def _floor(noise: float = NOISE, trials: int = 2000, seed: int = 1) -> NoiseFloor:
    """The noise floor of a scene as bright and as noisy as Cuprite."""
    return measure(
        _references(),
        WAVELENGTHS,
        noise=noise,
        brightness=BRIGHTNESS,
        trials=trials,
        seed=seed,
    )


def test_noise_fabricates_an_absorption_out_of_nothing() -> None:
    """The floor exists because a flat spectrum does not measure as flat."""
    floor = _floor()

    assert floor.depth > 0.0
    assert floor.angle < 90.0


def test_a_louder_sensor_fakes_a_deeper_absorption() -> None:
    assert _floor(noise=0.01).depth > _floor(noise=0.001).depth


def test_a_deep_clean_absorption_is_accepted() -> None:
    generator = np.random.default_rng(7)
    floor = _floor()
    pure = _dip(2210.0, 0.25)
    noisy = np.stack([
        pure + generator.normal(0.0, NOISE, pure.size) for _ in range(50)
    ])

    result = match(noisy, _references(), WAVELENGTHS)

    assert floor.accepts(result.depth, result.best_angle).all()


def test_a_flat_pixel_is_refused() -> None:
    """This is the pixel the floor was built from, so it must not pass."""
    generator = np.random.default_rng(11)
    floor = _floor()
    flat = BRIGHTNESS + generator.normal(0.0, NOISE, (200, WAVELENGTHS.size))

    result = match(flat, _references(), WAVELENGTHS)

    assert floor.accepts(result.depth, result.best_angle).mean() < 0.05


def test_a_mineral_that_is_not_in_the_list_is_refused() -> None:
    """A real, deep absorption that nothing in the list can explain.

    Depth alone would let this through, which is the whole reason the angle
    test is there as well.
    """
    floor = _floor()
    stranger = np.stack([_dip(2120.0, 0.30, width=8.0)])

    result = match(stranger, _references(), WAVELENGTHS)

    assert result.depth[0] > floor.depth
    assert not floor.accepts(result.depth, result.best_angle)[0]


def test_a_pixel_with_no_shape_at_all_is_refused() -> None:
    floor = _floor()
    featureless = np.stack([np.full(WAVELENGTHS.size, BRIGHTNESS)])

    result = match(featureless, _references(), WAVELENGTHS)

    assert not floor.accepts(result.depth, result.best_angle)[0]


def test_the_floor_is_the_same_on_every_run() -> None:
    assert _floor(seed=3).depth == pytest.approx(_floor(seed=3).depth)
    assert _floor(seed=3).angle == pytest.approx(_floor(seed=3).angle)
