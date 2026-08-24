"""Reference spectra from the USGS Spectral Library Version 7."""

from .splib07 import Spectrum, archive_size, fetch_archive, read_spectrum

__all__ = ["Spectrum", "archive_size", "fetch_archive", "read_spectrum"]
