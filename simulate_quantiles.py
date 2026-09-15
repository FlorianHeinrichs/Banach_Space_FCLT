#
# simulate_data.py
#
# Project: Functional Central Limit Theorem in Banach Spaces
# Date: 2026-02-13
# Author: Florian Heinrichs
#
# Script to simulate quantiles for self-normalization.

import numpy as np


def simulate_quantile(alpha: float, n_repetitions: int = 1000,
                      n_grid: int = 1000) -> float:
    """
    Simulate (1 - alpha) quantile of limiting distribution.

    :param alpha: Level of quantile.
    :param n_repetitions: Number of repetitions.
    :param n_grid: Number of grid points per Brownian motion.
    :return: (1 - alpha) quantile of limiting distribution.
    """
    realizations = simulate_ratio(n_repetitions, n_grid)
    quantile = np.sort(realizations)[int(n_repetitions * (1 - alpha))]

    return quantile


def simulate_ratio(n_repetitions: int = 1000,
                   n_grid: int = 1000) -> np.ndarray:
    """
    Simulate realizations of limiting distribution.

    :param n_repetitions: Number of repetitions.
    :param n_grid: Number of grid points per Brownian motion.
    :return: (1 - alpha) quantile of limiting distribution.
    """
    bm = simulate_brownian_motions(n_repetitions, n_grid)

    indices = np.arange(1, n_grid + 1) / n_grid
    numerator = bm - indices[None, :] * bm[:, -1:]

    i, k = indices[None, :, None], indices[None, None, :]
    bmi, bmk = bm[..., None], bm[:, None, :]
    mask = i <= k
    first_int = np.sum(((bmi - (i / k) * bmk) * mask) ** 2, axis=1) / n_grid
    bmi, bmk = (bm[:, -1:] - bm)[..., None], (bm[:, -1:] - bm)[:, None, :]
    second_int = np.zeros_like(bm)
    second_int[:, :-1] = np.sum(((bmi - (1 - i) / (1 - k[..., :-1]) * bmk[..., :-1])
                                 * (1 - mask[..., :-1])) ** 2, axis=1) / n_grid
    denominator = np.sqrt(first_int + second_int)

    realizations = np.max(np.abs(numerator) / denominator, axis=1)

    return realizations


def simulate_brownian_motions(n_repetitions: int = 1000,
                              n_grid: int = 1000) -> np.ndarray:
    """
    Simulate trajectories of (independent) Brownian motions.

    :param n_repetitions: Number of trajectories.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :return: Simulated trajectories as NumPy array.
    """
    rng = np.random.default_rng()

    increments = (rng.standard_normal(size=(n_repetitions, n_grid))
                  / np.sqrt(n_grid))
    increments[:, 0] = 0
    W = np.cumsum(increments, axis=-1)

    return W
