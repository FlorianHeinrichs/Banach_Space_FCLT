#
# simulate_data.py
#
# Project: Functional Central Limit Theorem in Banach Spaces
# Date: 2026-02-13
# Author: Florian Heinrichs
#
# Main script to conduct experiments.

from datetime import datetime
import json

import numpy as np

from self_normalization import self_normalization_test
from simulate_data import generate_errors
from simulate_quantiles import simulate_quantile


def experiment(n_time: int, error_type: str, quantile: float, height: float,
               b: np.ndarray, n_space: int = 100,
               n_samples: int = 1000) -> np.ndarray:
    """
    Compare MAE and MSE of the three estimators (NWE, LLE, Jackknife) for mu and
    its derivative.

    :param n_time: Number of time points.
    :param error_type: Error type, either of 'IID_BM', 'IID_BB', 'FAR_BM',
        'FAR_BB', 'Heteroscedastic_Independent', 'Heteroscedastic_Dependent_1',
        'Heteroscedastic_Dependent_2'.
    :param quantile: Quantile used for test.
    :param height: Height of the change simulation.
    :param b: "Functional" for projection of functional data to R. Given as
        NumPy array with shape (n_space,).
    :param n_space: Number of spatial (grid) points.
    :param n_samples: Number of simulated time series per model.
    :return: Test decisions as NumPy array of length n_samples.
    """
    mu = np.zeros((n_time, n_space))
    mu[n_time // 2:] = np.concatenate([np.ones(n_space // 2),
                                       -np.ones(n_space - n_space // 2)])
    errors = generate_errors(error_type, n_time, n_space, n_samples)
    X = height * mu[None, ...] + errors

    test_decisions = self_normalization_test(X, b, quantile)

    return test_decisions


def get_config(n_samples: int, n_space: int, quantile: float) -> list:
    """
    Get list of configurations for experiments.

    :param n_samples: Number of generated time series.
    :param n_space: Number of spatial (grid) points.
    :param quantile: Quantile used for test.
    :return: List of configurations, where each entry of the list is a tuple
        of arguments of experiment().
    """
    errors = ['IID_BM', 'IID_BB', 'FAR_BM', 'FAR_BB',
              'Heteroscedastic_Independent', 'Heteroscedastic_Dependent_1',
              'Heteroscedastic_Dependent_2']
    bs = [np.ones(n_space) / n_space,
          np.concatenate([np.ones(n_space // 2),
                          -np.ones(n_space - n_space // 2)]) / n_space]
    heights = np.concatenate([np.arange(0, 1, 0.1), np.arange(1, 11)])
    config = [
        ((n, error_type, quantile, height, b), {'n_space': n_space, 'n_samples': n_samples})
        for n in [100, 200, 500]
        for b in bs
        for error_type in errors
        for height in heights
    ]

    return config


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return np.array(data).tolist()
    elif isinstance(data, tuple):
        return [convert_to_serializable(value) for value in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


if __name__ == '__main__':
    n_samples = 1000
    n_space = 100
    alpha = 0.05
    results = {}

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"../results/samples{n_samples}_" + now + ".json"

    quantile = simulate_quantile(alpha, n_repetitions=100000, n_grid=100)
    config = get_config(n_samples, n_space, quantile)

    for args, kwargs in config:
        arg_str = (f"n{args[0]}_{args[1]}_height{args[3]:.1f}_"
                   f"b{np.sum(args[4]) + 1e-5:.0f}")
        print(f"Starting experiment: {arg_str}")
        results[arg_str] = experiment(*args, **kwargs)

    print(results)
