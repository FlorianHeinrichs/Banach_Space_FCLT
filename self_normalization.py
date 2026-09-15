#
# self_normalization.py
#
# Project: Functional Central Limit Theorem in Banach Spaces
# Date: 2026-02-12
# Author: Florian Heinrichs
#
# Implementation of the proposed self-normalization procedure for change point
# detection.

import numpy as np

from simulate_quantiles import simulate_ratio


def self_normalization_test(X: np.ndarray, b: np.ndarray,
                            quantile: float,
                            return_p_value: bool = False) -> np.ndarray | tuple:
    """
    Conduct change point test based on self-normalization in the AMOC ("At Most
    One Change") setting.

    :param X: Functional observations, given as NumPy array with shape
        (n_time, n_space), where n_time is the number of functional
        observations and n_space is the number of grid points.
        If multiple time series are provided simultaneously, the shape is
        (n_ts, n_time, n_space).
    :param b: "Functional" for projection of functional data to R. Given as
        NumPy array with shape (n_space,).
    :param return_p_value: Whether to return p-value or not.
    :param quantile: Quantile used for decision rule.
    :return: Test decision as bool.
    """
    expand_dims = len(X.shape) == 2

    if expand_dims:
        X = X[None, ...]

    n_time = X.shape[1]
    X_proj = X @ b
    T_n = np.cumsum(X_proj, axis=1) / n_time

    indices = np.arange(1, n_time + 1)
    numerator = np.sqrt(n_time) * (T_n - indices[None, :] / n_time * T_n[:, -1:])

    i, k = indices[None, :, None], indices[None, None, :]
    Ti, Tk = T_n[..., None], T_n[:, None, :]
    mask = i <= k
    first_sum = np.sum(((Ti - (i / k) * Tk) * mask) ** 2, axis=1)
    Ti, Tk = (T_n[:, -1:] - T_n)[..., None], (T_n[:, -1:] - T_n)[:, None, :]
    second_sum = np.zeros_like(T_n)
    second_sum[:, :-1] = np.sum(
        ((Ti - (n_time - i) / (n_time - k[..., :-1]) * Tk[..., :-1])
         * (1 - mask[..., :-1])) ** 2, axis=1
    )

    denominator = np.sqrt(first_sum + second_sum)

    test_statistic = np.max(np.abs(numerator) / denominator, axis=1)
    test_decision = test_statistic > quantile

    if return_p_value:
        ratio = simulate_ratio(n_repetitions=10000, n_grid=100)
        p_value = np.mean(ratio[:, None] > test_statistic[None, :], axis=0)

    if expand_dims and return_p_value:
        return test_decision[0], p_value[0]
    elif expand_dims:
        return test_decision[0]
    elif return_p_value:
        return test_statistic, p_value
    else:
        return test_decision

