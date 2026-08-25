"""Reference spectra from the USGS Spectral Library Version 7."""

from .continuum import deepest_feature, remove
from .mapping import Mapped, group_indices, map_scene
from .matching import Match, match
from .rejection import NoiseFloor, measure
from .resample import convolve, extra_blur
from .splib07 import Spectrum, archive_size, fetch_archive, read_spectra, read_spectrum

__all__ = [
    "Mapped",
    "Match",
    "NoiseFloor",
    "Spectrum",
    "archive_size",
    "convolve",
    "deepest_feature",
    "extra_blur",
    "fetch_archive",
    "group_indices",
    "map_scene",
    "match",
    "measure",
    "read_spectra",
    "read_spectrum",
    "remove",
]
