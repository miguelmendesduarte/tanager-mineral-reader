"""The spectral axis of a cube."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import NoBandsInRangeError


@dataclass(frozen=True)
class Wavelengths:
    """Band centres, band widths, and which bands are worth using.

    Attributes:
        centres: Centre wavelength of each band, in nanometres.
        widths: Full width at half maximum of each band, in nanometres.
        usable: Whether each band carries usable signal. The reflectance
            product flags the two ranges where atmospheric water vapour
            absorbs almost everything; the radiance product does not, in
            which case every band is reported as usable.
    """

    centres: NDArray[np.float64]
    widths: NDArray[np.float64]
    usable: NDArray[np.bool_]

    def __len__(self) -> int:
        """Number of bands."""
        return int(self.centres.size)

    def indices_between(
        self,
        low: float,
        high: float,
        *,
        usable_only: bool = True,
    ) -> NDArray[np.intp]:
        """Select the bands whose centre falls inside a wavelength range.

        Args:
            low: Lower bound in nanometres, included.
            high: Upper bound in nanometres, included.
            usable_only: Skip the bands the file flags as unusable.

        Returns:
            NDArray[np.intp]: Indices of the selected bands, in order.

        Raises:
            NoBandsInRangeError: If the range holds no selectable band.
        """
        selected = (self.centres >= low) & (self.centres <= high)
        if usable_only:
            selected &= self.usable

        indices = np.flatnonzero(selected)
        if indices.size == 0:
            raise NoBandsInRangeError(low, high)
        return indices
