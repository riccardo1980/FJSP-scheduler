from typing import Tuple

import numpy as np
import pytest

from demo_scheduler import crossover
from tests import corner_cases

# fmt: off
# T1: Figure 5 example
UNIFORM_T1P0 = np.asarray([ 1, 13, 18, 22, 12,  5,  7, 25])  # noqa: E201, E222, E241
UNIFORM_T1P1 = np.asarray([ 3, 15, 16, 25, 11,  8, 10, 18])  # noqa: E201, E222, E241
UNIFORM_T1PR = np.asarray([0.23, 0.65, 0.31, 0.11, 0.63, 0.78, 0.47, 0.56])  # noqa: E201, E222, E501, E241
UNIFORM_T1O0 = np.asarray([ 3, 13, 16, 25, 12,  5, 10, 25])  # noqa: E201, E222, E241
UNIFORM_T1O1 = np.asarray([ 1, 15, 18, 22, 11,  8,  7, 18])  # noqa: E201, E222, E241
# fmt: on


@pytest.mark.parametrize(
    "parents, pr, expected",
    [((UNIFORM_T1P0, UNIFORM_T1P1), UNIFORM_T1PR, (UNIFORM_T1O0, UNIFORM_T1O1))],
)
def test_uniform(
    parents: Tuple[np.ndarray, np.ndarray],
    pr: np.ndarray,
    expected: Tuple[np.ndarray, np.ndarray],
) -> None:
    got = crossover.__uniform(parents=parents, pr=pr)
    for o_got, o_expected in zip(got, expected):
        np.testing.assert_array_equal(o_got, o_expected)


# fmt: off
# T1: Figure 7 - subportion 0 example
SUB_MOC_T1P0 =   np.asarray([1, 4, 2, 3])  # noqa: E201, E222
SUB_MOC_T1P1 =   np.asarray([3, 2, 1, 4])  # noqa: E201, E222
SUB_MOC_T1PRES = np.asarray([1, 0, 1, 1]).astype(bool)  # noqa: E201, E222
SUB_MOC_T1O =    np.asarray([1, 4, 2, 3])  # noqa: E201, E222

# T2: Figure 7 - subportion 1 example
SUB_MOC_T2P0 =   np.asarray([8, 6, 5, 7])  # noqa: E201, E222
SUB_MOC_T2P1 =   np.asarray([6, 7, 8, 5])  # noqa: E201, E222
SUB_MOC_T2PRES = np.asarray([0, 1, 1, 0]).astype(bool)  # noqa: E201, E222
SUB_MOC_T2O =    np.asarray([7, 6, 5, 8])  # noqa: E201, E222
# fmt: on


@pytest.mark.parametrize(
    "parents, is_preserved, expected",
    [
        ((SUB_MOC_T1P0, SUB_MOC_T1P1), SUB_MOC_T1PRES, SUB_MOC_T1O),
        ((SUB_MOC_T2P0, SUB_MOC_T2P1), SUB_MOC_T2PRES, SUB_MOC_T2O),
    ],
)
def test_subportion_moc(
    parents: Tuple[np.ndarray, np.ndarray],
    is_preserved: np.ndarray,
    expected: np.ndarray,
) -> None:
    got = crossover.__subportion_moc(parents=parents, is_preserved=is_preserved)
    np.testing.assert_array_equal(got, expected)


# fmt: off
# T1: Figure 7 example
SINGLE_MOC_T1P0   = np.asarray([1, 4, 2, 3, 8, 6, 5, 7])  # noqa: E201, E221, E222
SINGLE_MOC_T1P1   = np.asarray([3, 2, 1, 4, 6, 7, 8, 5])  # noqa: E201, E221, E222
SINGLE_MOC_T1PRES_IDX = np.asarray([0, 2, 3, 5, 6])  # noqa: E201, E221, E222
SINGLE_MOC_T1O    = np.asarray([1, 4, 2, 3, 7, 6, 5, 8])  # noqa: E201, E221, E222
SINGLE_MOC_T1SUBSTART = np.asarray([0, 4])  # noqa: E201, E221, E222
# fmt: on


@pytest.mark.parametrize(
    "parents, preserved_genes_indices, subportions_start, expected",
    [
        (
            (SINGLE_MOC_T1P0, SINGLE_MOC_T1P1),
            SINGLE_MOC_T1PRES_IDX,
            SINGLE_MOC_T1SUBSTART,
            SINGLE_MOC_T1O,
        )
    ],
)
def test_single_offspring_moc(
    parents: Tuple[np.ndarray, np.ndarray],
    preserved_genes_indices: np.ndarray,
    subportions_start: np.ndarray,
    expected: np.ndarray,
) -> None:
    got = crossover.__single_offspring_moc(
        parents=parents,
        preserved_genes_indices=preserved_genes_indices,
        subportions_start=subportions_start,
    )
    np.testing.assert_array_equal(got, expected)


def test_choose_from_0() -> None:
    corner_cases.choose_from_0()


def test_choose_from_1() -> None:
    corner_cases.choose_from_1()


def test_no_swap() -> None:
    corner_cases.no_swap()


def test_all_swap() -> None:
    corner_cases.all_swap()


def test_single_subportion() -> None:
    corner_cases.single_subportion()
