import logging

import numpy as np

from demo_scheduler import crossover


def choose_from_0() -> None:
    logging.info(
        "----------- MOC crossover: all genes are chosen from parent 0 -----------"
    )
    p0 = np.asarray([1, 4, 2, 3, 8, 6, 5, 7])  # right segment
    p1 = np.asarray([3, 2, 1, 4, 6, 7, 8, 5])  # right segment

    subportions_start = np.asarray([0, 4])
    preserved_genes_indices = np.asarray(range(p0.size))

    offspring = crossover.__single_offspring_moc(
        (p0, p1),
        preserved_genes_indices=preserved_genes_indices,
        subportions_start=subportions_start,
    )

    np.testing.assert_array_equal(offspring, p0)


def choose_from_1() -> None:
    logging.info(
        "----------- MOC crossover: all genes are chosen from parent 1 -----------"
    )
    p0 = np.asarray([1, 4, 2, 3, 8, 6, 5, 7])  # right segment
    p1 = np.asarray([3, 2, 1, 4, 6, 7, 8, 5])  # right segment

    subportions_start = np.asarray([0, 4])
    preserved_genes_indices = np.asarray([])

    offspring = crossover.__single_offspring_moc(
        (p0, p1),
        preserved_genes_indices=preserved_genes_indices,
        subportions_start=subportions_start,
    )

    np.testing.assert_array_equal(offspring, p1)


def no_swap() -> None:
    logging.info(
        "----------- uniform crossover: no gene values are swapped -----------"
    )
    p0 = np.asarray([1, 13, 18, 22, 12, 5, 7, 25])  # left segment
    p1 = np.asarray([3, 15, 16, 25, 11, 8, 10, 18])  # left segment

    fill_value = 0.6
    logging.info(f"generating left segment offsprings forcing all pr == {fill_value}")
    offsprings = crossover.__uniform((p0, p1), np.full(p0.shape, fill_value=fill_value))

    logging.debug(f"P0: {p0}")
    logging.debug(f"P1: {p1}")
    logging.debug(f"O0: {offsprings[0]}")
    logging.debug(f"O1: {offsprings[1]}")

    np.testing.assert_array_equal(offsprings[0], p0)
    np.testing.assert_array_equal(offsprings[1], p1)


def all_swap() -> None:
    logging.info(
        "----------- uniform crossover: all gene values are swapped -----------"
    )
    p0 = np.asarray([1, 13, 18, 22, 12, 5, 7, 25])  # left segment
    p1 = np.asarray([3, 15, 16, 25, 11, 8, 10, 18])  # left segment

    fill_value = 0.4
    logging.info(f"generating left segment offsprings forcing all pr == {fill_value}")
    offsprings = crossover.__uniform((p0, p1), np.full(p0.shape, fill_value=fill_value))

    logging.debug(f"P0: {p0}")
    logging.debug(f"P1: {p1}")
    logging.debug(f"O0: {offsprings[0]}")
    logging.debug(f"O1: {offsprings[1]}")

    np.testing.assert_array_equal(offsprings[0], p1)
    np.testing.assert_array_equal(offsprings[1], p0)


def single_subportion() -> None:
    logging.info("----------- MOC crossover: a single subportion -----------")
    p0 = np.asarray([1, 4, 2, 3, 8, 6, 5, 7])  # right segment
    p1 = np.asarray([3, 2, 1, 4, 6, 7, 8, 5])  # right segment

    subportions_start = np.asarray([0])
    preserved_genes_indices = np.asarray([0, 1, 7])
    expected = np.asarray([1, 4, 3, 2, 6, 8, 5, 7])

    logging.info("generating right segment offsprings")
    offspring = crossover.__single_offspring_moc(
        (p0, p1),
        preserved_genes_indices=preserved_genes_indices,
        subportions_start=subportions_start,
    )

    np.testing.assert_array_equal(offspring, expected)
