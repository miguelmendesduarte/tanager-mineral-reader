"""Comparing two maps of the same ground.

Mapping a place twice from two separate passes and asking how often the two
answers match is the only test of this method that needs no ground truth. It
does not show the maps are right — a mistake made consistently is made twice —
but it shows which parts of the answer are stable and which are a coin toss,
and those are different claims worth separating.

Two conditions have to hold before the question means anything. The passes must
be genuinely separate, because a single pass delivered in two pieces would
agree with itself perfectly and prove nothing. And the maps must be compared
where both actually looked, on the same pixels, which the shared grid makes a
matter of taking a window rather than resampling.

The word used throughout is agreement rather than accuracy. Neither map is
truth, and where they differ, nothing here says which one was wrong.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .core.exceptions import NoOverlapError, NotIndependentError
from .spectra.mapping import UNCLASSIFIED, Mapped


@dataclass(frozen=True)
class Agreement:
    """How two maps of the same ground compare.

    Attributes:
        matrix: Counts of every combination of the two maps' labels, shaped
            (groups, groups), the first map down and the second across. Called
            a concordance matrix rather than a confusion matrix: a confusion
            matrix has a truth axis, and neither of these is truth.
        groups: Names of the mineral groups, in matrix order.
        compared: Pixels where both maps named a mineral.
        overlapping: Pixels where both maps looked, named or not.
    """

    matrix: NDArray[np.int64]
    groups: tuple[str, ...]
    compared: int
    overlapping: int

    @property
    def agreed(self) -> int:
        """Pixels both maps gave the same name."""
        return int(np.trace(self.matrix))

    @property
    def rate(self) -> float:
        """Share of compared pixels the two maps agreed on."""
        return self.agreed / self.compared if self.compared else float("nan")

    @property
    def kappa(self) -> float:
        """Agreement above what the two maps would reach by chance.

        Raw agreement flatters a map dominated by one class: label everything
        muscovite twice over and it reads as perfect. Cohen's kappa subtracts
        the agreement the two label distributions would produce if they were
        independent, so 0 is chance and 1 is exact.
        """
        if not self.compared:
            return float("nan")
        observed = self.rate
        rows = self.matrix.sum(axis=1) / self.compared
        columns = self.matrix.sum(axis=0) / self.compared
        expected = float(rows @ columns)
        return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0

    def rows(self) -> list[tuple[str, list[int], int]]:
        """The matrix as named rows, leaving out groups neither map used.

        Returns:
            list[tuple[str, list[int], int]]: For each group either map named,
                its counts against every such group and its row total.
        """
        used = [
            index
            for index in range(len(self.groups))
            if self.matrix[index].sum() or self.matrix[:, index].sum()
        ]
        return [
            (
                self.groups[index],
                [int(self.matrix[index, other]) for other in used],
                int(self.matrix[index].sum()),
            )
            for index in used
        ]

    def columns(self) -> list[str]:
        """Names of the groups the matrix rows are counted against."""
        return [
            self.groups[index]
            for index in range(len(self.groups))
            if self.matrix[index].sum() or self.matrix[:, index].sum()
        ]

    def per_group(self) -> dict[str, float]:
        """Share of each group's pixels the other map agreed on.

        Averaged over both directions, since neither map is the reference.
        """
        shares = {}
        for index, group in enumerate(self.groups):
            total = int(self.matrix[index].sum() + self.matrix[:, index].sum())
            if total:
                shares[group] = float(2 * self.matrix[index, index] / total)
        return shares


def compare(
    first: Mapped, second: Mapped, *, strips: tuple[str | None, str | None]
) -> Agreement:
    """Compare two maps over the ground they share.

    Args:
        first: A mineral map.
        second: Another, of overlapping ground on the same grid.
        strips: The pass each map came from, to check they are separate.

    Returns:
        Agreement: How the two compare where both looked.

    Raises:
        NotIndependentError: If both maps came from the same pass.
        GridsNotAlignedError: If the two grids do not share a lattice.
        NoOverlapError: If they do not overlap at all.
    """
    _reject_same_pass(strips)
    left, right = _shared(first, second)

    both_named = (left != UNCLASSIFIED) & (right != UNCLASSIFIED)
    groups = first.groups
    matrix = np.zeros((len(groups), len(groups)), dtype=np.int64)
    np.add.at(matrix, (left[both_named], right[both_named]), 1)

    agreement = Agreement(
        matrix=matrix,
        groups=groups,
        compared=int(both_named.sum()),
        overlapping=int(left.size),
    )
    logger.info(
        "{} pixels of shared ground, {} named by both, {:.1f}% agreed, kappa {:.2f}",
        agreement.overlapping,
        agreement.compared,
        100 * agreement.rate,
        agreement.kappa,
    )
    return agreement


def by_confidence(
    first: Mapped,
    second: Mapped,
) -> dict[str, tuple[float, int]]:
    """Agreement split by whether each map settled its own ranking.

    The test the confidence layer has to pass. A pixel both maps called
    settled should agree far more often than one where either was undecided;
    if it does not, the confidence means nothing and should not be published
    as though it did.

    Args:
        first: A mineral map.
        second: Another, of overlapping ground on the same grid.

    Returns:
        dict[str, tuple[float, int]]: Agreement rate and pixel count, for
            pixels both maps settled and for the rest.
    """
    left, right = _shared(first, second)
    left_firm, right_firm = _shared_layer(first, second, "resolved")

    both_named = (left != UNCLASSIFIED) & (right != UNCLASSIFIED)
    firm = both_named & left_firm.astype(bool) & right_firm.astype(bool)
    loose = both_named & ~firm

    return {
        "both settled": _rate(left, right, firm),
        "either unsettled": _rate(left, right, loose),
    }


def _rate(
    left: NDArray[np.intp],
    right: NDArray[np.intp],
    selected: NDArray[np.bool_],
) -> tuple[float, int]:
    """Agreement rate and pixel count over a selection."""
    count = int(selected.sum())
    if not count:
        return float("nan"), 0
    return float((left[selected] == right[selected]).mean()), count


def _reject_same_pass(strips: tuple[str | None, str | None]) -> None:
    """Refuse two pieces of one pass, which would agree with themselves."""
    first, second = strips
    if first is not None and first == second:
        raise NotIndependentError(first)


def _shared(first: Mapped, second: Mapped) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The two maps' labels over the ground they both cover."""
    left, right = _shared_layer(first, second, "labels")
    return left.astype(np.intp), right.astype(np.intp)


def _shared_layer(
    first: Mapped,
    second: Mapped,
    name: str,
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    """One layer of each map, cut to the ground they both cover.

    Raises:
        GridsNotAlignedError: If the two grids do not share a lattice.
        NoOverlapError: If they do not overlap at all.
    """
    row, column = first.grid.offset_to(second.grid)
    first_rows, first_columns = first.grid.shape
    second_rows, second_columns = second.grid.shape

    top, left = max(row, 0), max(column, 0)
    bottom = min(first_rows, row + second_rows)
    right = min(first_columns, column + second_columns)
    if bottom <= top or right <= left:
        raise NoOverlapError

    return (
        getattr(first, name)[top:bottom, left:right],
        getattr(second, name)[top - row : bottom - row, left - column : right - column],
    )
