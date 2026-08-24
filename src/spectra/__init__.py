"""Reference spectra from the USGS Spectral Library Version 7."""

from .resample import blunter_bands, convolve
from .splib07 import Spectrum, archive_size, fetch_archive, read_spectra, read_spectrum

__all__ = [
    "Spectrum",
    "archive_size",
    "blunter_bands",
    "convolve",
    "fetch_archive",
    "read_spectra",
    "read_spectrum",
]
