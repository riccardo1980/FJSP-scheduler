import logging
import random

import numpy as np

from demo_scheduler import crossover


def run_random() -> None:
    """
    Random run

    Testing a number of seeds, given fixed parents, subportions and MOC_rate
    """

    logging.info("########################## Random run ##########################")

    seeds = [4, 8, 15, 16, 23, 42]
    p0 = (
        np.asarray([1, 13, 18, 22, 12, 5, 7, 25]),  # left segment
        np.asarray([1, 4, 2, 3, 8, 6, 5, 7]),  # right segment
    )
    p1 = (
        np.asarray([3, 15, 16, 25, 11, 8, 10, 18]),  # left segment
        np.asarray([3, 2, 1, 4, 6, 7, 8, 5]),  # right segment
    )
    subportions_start = np.asarray([0, 4])
    MOC_rate = 0.4

    for it, seed in enumerate(seeds):
        logging.info(f"-------------- Run {it} -----------------")
        random.seed(seed)
        logging.info("generating left segment offsprings")
        left_segment_offsprings = crossover.uniform((p0[0], p1[0]))

        logging.info("generating right segment offsprings")
        right_segment_offsprings = crossover.moc(
            (p0[1], p1[1]), MOC_rate=MOC_rate, subportions_start=subportions_start
        )

        # first and second offsprings will be composed as:
        _ = left_segment_offsprings[0], right_segment_offsprings[0]
        _ = left_segment_offsprings[1], right_segment_offsprings[1]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(module)s:%(funcName)s | %(message)s",
    )

    run_random()
