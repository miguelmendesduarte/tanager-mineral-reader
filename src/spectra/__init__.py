"""Reference spectra from the USGS Spectral Library Version 7."""

from .continuum import deepest_feature, remove
from .resample import convolve, extra_blur
from .splib07 import Spectrum, archive_size, fetch_archive, read_spectra, read_spectrum

__all__ = [
    "Spectrum",
    "archive_size",
    "convolve",
    "deepest_feature",
    "extra_blur",
    "fetch_archive",
    "read_spectra",
    "read_spectrum",
    "remove",
]
