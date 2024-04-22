import logging
import random
from copy import deepcopy
from typing import Tuple

import numpy as np


def __uniform(
    parents: Tuple[np.ndarray, np.ndarray], pr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    INTERNAL Uniform crossover

    If pr > 0.5
        offspring 0 takes the gene from parent 0
        offspring 1 takes the gene from parent 1
    otherwise
        offspring 0 takes the gene from parent 1
        offspring 1 takes the gene from parent 0

    :param parents: tuple of left segment of parents
    :param pr: probabilities
    :return: tuple of left segment of offsprings

    """
    THRESHOLD = 0.5

    # we do swap if pr_i <= than THRESHOLD
    mask = pr <= THRESHOLD
    logging.debug(f"do we swap?: {mask}")

    # we copy if pr_i > than THRESHOLD
    off_0 = deepcopy(parents[0])
    off_1 = deepcopy(parents[1])

    off_0[mask] = parents[1][mask]
    off_1[mask] = parents[0][mask]

    return (off_0, off_1)


def uniform(parents: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover

    Draws a probability array and calls __uniform for uniform crossover implementation

    memory:
        - 2 temporary array of same size as gene input array

    complexity:
        - linear on gene length

    :param parents: tuple of left segment of parents
    :param pr: probabilities
    :return: tuple of left segment of offsprings

    """
    # uniform in [0,1)
    pr = np.random.rand(*parents[0].shape)
    logging.debug(f"pr: {pr}")

    offsprings = __uniform(parents=parents, pr=pr)
    logging.debug(f"P0: {parents[0]}")
    logging.debug(f"P1: {parents[1]}")
    logging.debug(f"O0: {offsprings[0]}")
    logging.debug(f"O1: {offsprings[1]}")

    return offsprings


def __subportion_moc(
    parents: Tuple[np.ndarray, np.ndarray], is_preserved: np.ndarray
) -> np.ndarray:
    """
    INTERNAL subportion-wide Modified Order Crossover (MOC)

    Genes selected by mask is_preserved are chosen from p0,
    remaining values are extracted from p1, preserving their relative order

    :param parents: tuple of right subportions of parents
    :param is_preserved: boolean array, true when gene value is preserved
    :return: tuple of right subportions of offsprings

    """
    # values from P0
    P0_values = parents[0][is_preserved]

    # mask of genes values coming from P1
    is_from_P1 = ~np.isin(parents[1], P0_values)

    offspring = np.zeros_like(parents[0])

    # passing genes positions and values from P0
    offspring[is_preserved] = parents[0][is_preserved]

    # adding remaining values from P1 preserving relative order
    offspring[~is_preserved] = parents[1][is_from_P1]

    return offspring


def __single_offspring_moc(
    parents: Tuple[np.ndarray, np.ndarray],
    preserved_genes_indices: np.ndarray,
    subportions_start: np.ndarray,
) -> np.ndarray:
    """
    INTERNAL Segment-wide Modified Order Crossover (MOC)

    Creates a mask of same size as genes input array containing True
    if gene value has to be preserved from p0, False otherwise
    For each subportion, calls __subportion_moc with subportions of parents and mask

    :param parents: tuple of right segment of parents
    :param preserved_genes_indices: indices of preserved genes from parent 0
    :return: right segment of offspring

    """
    logging.debug(f"Subportions start: {subportions_start}")
    logging.debug(f"Preserved genes indices: {preserved_genes_indices}")
    logging.debug(f"P0: {parents[0]}")
    logging.debug(f"P1: {parents[1]}")

    is_preserved = np.zeros_like(parents[0]).astype(bool)
    if any(preserved_genes_indices):
        is_preserved[preserved_genes_indices] = True

    logging.debug(f"Preserved mask: {is_preserved}")

    offspring = np.zeros_like(parents[0])

    n = parents[0].size
    for sp_start, sp_end in zip(subportions_start, np.append(subportions_start[1:], n)):
        sub_parent_0 = parents[0][sp_start:sp_end]
        sub_parent_1 = parents[1][sp_start:sp_end]
        sub_is_preserved = is_preserved[sp_start:sp_end]
        offspring[sp_start:sp_end] = __subportion_moc(
            (sub_parent_0, sub_parent_1), sub_is_preserved
        )

    logging.debug(f" O: {offspring}")

    return offspring


def moc(
    parents: Tuple[np.ndarray, np.ndarray],
    MOC_rate: float,
    subportions_start: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment-wide Modified Order Crossover

    Selects randomly a MOC_reate portion of genes,
    then calls __single_offspring_moc for one offspring generation

    Procedure is repeated with reverted roles between parents

    memory:
        - 2 temporary array of same size as gene input array
        - 4 temporary arrays of max size the same as max subportion size

    complexity:
        - linear on gene length

    :param parents: tuple of right segment of parents
    :param MOC_rate: fraction of genes to preserve
    :param subportions_start: indices where a new subportion starts,
        use [0] when having a single subportion
    :return: tuple of right segment of offsprings

    """
    # determine the number of genes to preserve
    n = parents[0].size
    total_preserved_genes = int(round(MOC_rate * n))
    logging.debug(f"Total preserved genes: {total_preserved_genes} over {n}")

    logging.debug("Offspring 0")
    preserved_genes_indices = np.asarray(random.sample(range(n), total_preserved_genes))

    offspring_0 = __single_offspring_moc(
        parents=parents,
        preserved_genes_indices=preserved_genes_indices,
        subportions_start=subportions_start,
    )

    logging.debug("Offspring 1")
    preserved_genes_indices = np.asarray(random.sample(range(n), total_preserved_genes))

    offspring_1 = __single_offspring_moc(
        parents=(parents[1], parents[0]),
        preserved_genes_indices=preserved_genes_indices,
        subportions_start=subportions_start,
    )

    return (offspring_0, offspring_1)
