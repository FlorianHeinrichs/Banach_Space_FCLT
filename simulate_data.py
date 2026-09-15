#
# simulate_data.py
#
# Project: Functional Central Limit Theorem in Banach Spaces
# Date: 2026-02-13
# Author: Florian Heinrichs
#
# Script to generate simulated data.
# Error processes correspond to models (N_1), (N_2), (N_4), (A_3) -- (A_6) of
# Bücher, A., Dette, H., & Heinrichs, F. (2023). A portmanteau-type test for
# detecting serial correlation in locally stationary functional time series.
# Statistical Inference for Stochastic Processes, 26(2), 255-278.

from typing import Callable

import numpy as np


def generate_errors(error_type: str, n_time: int, n_space: int,
                    n_repetitions: int) -> np.ndarray:
    """
    Wrapper function to generate specified errors.

    :param error_type: Type of error.
    :param n_time: Length of time series.
    :param n_space: Spatial resolution of each function.
    :param n_repetitions: Number of generated time series.
    :return: Generated time series as NumPy array of shape:
        (n_repetitions, n_time, n_space).
    """
    error_funcs = {
        'IID_BM': (generate_iid, {'innovation_type': 'BM'}),
        'IID_BB': (generate_iid, {'innovation_type': 'BB'}),
        'FAR_BM': (generate_far, {'innovation_type': 'BM'}),
        'FAR_BB': (generate_far, {'innovation_type': 'BB'}),
        'Heteroscedastic_Independent': (generate_hetero_indep, {}),
        'Heteroscedastic_Dependent_1': (generate_hetero_dep, {'model': 1}),
        'Heteroscedastic_Dependent_2': (generate_hetero_dep, {'model': 2}),
    }

    error_func, error_kwargs = error_funcs[error_type]
    errors = np.stack([error_func(n_time, n_space, **error_kwargs)
                       for _ in range(n_repetitions)])

    return errors


def brownian_motion(n_space: int) -> np.ndarray:
    """
    Simulate one path of a Brownian motion on [0,1].

    :param n_space: Number of grid points.
    :return: Trajectory of Brownian motion.
    """
    dt = 1 / n_space
    increments = np.sqrt(dt) * np.random.randn(n_space)
    bm = np.cumsum(increments)

    return bm


def brownian_bridge(n_space: int) -> np.ndarray:
    """
    Simulate one path of a Brownian bridge on [0,1].

    :param n_space: Number of grid points.
    :return: Trajectory of Brownian bridge.
    """
    bm = brownian_motion(n_space)
    bb = bm - np.linspace(0, 1, n_space) * bm[-1]

    return bb


def generate_iid(n_time: int, n_space: int, innovation_type: str) -> np.ndarray:
    """
    Generate functional time series of i.i.d. Brownian motions.

    :param n_time: Number of time points.
    :param n_space: Number of spatial points.
    :param innovation_type: Type of innovations, either of 'BM', 'BB', 'IID'.
    :return: Functional time series of i.i.d. processes.
    """
    if innovation_type == 'BM':
        func = brownian_motion
    elif innovation_type == 'BB':
        func = brownian_bridge
    elif innovation_type == 'IID':
        func = np.random.randn
    else:
        raise ValueError(f"{innovation_type} is not a valid innovation type.")

    noise = np.array([func(n_space) for _ in range(n_time)])

    return noise


def apply_rho(f: np.ndarray, n_space: int, kernel: Callable) -> np.ndarray:
    """
    Apply the integral operator ρ to a function f defined on grid,
    where (ρ f)(τ) = ∫₀¹ K(τ,σ) f(σ)dσ.

    The kernel K is a function K(tau, sigma).
    We use the trapezoidal rule to approximate the integral.

    Parameters:
    :param f: Function values on an equidistant grid of the interval [0, 1]
    :param n_space: Number of grid points.
    :param kernel: Kernel function.
    :return: Resulting function ρ f
    """
    x_space = np.linspace(0, 1, n_space)

    result = np.zeros_like(f)
    for i, tau in enumerate(x_space):
        integrand = kernel(tau, x_space) * f
        result[i] = np.mean(integrand)

    return result


def generate_far(n_time: int,
                 n_space: int,
                 innovation_type: str = 'BM') -> np.ndarray:
    """
    Simulate a FAR(1) process defined by: X_t = ρ(X_{t-1}) + ε_t, where the
    integral operator ρ is defined via the kernel
    K(tau, sigma) = c_w * min(tau, sigma), and ε_t is a centered innovation
    drawn either as a Brownian motion (BM) or a Brownian bridge (BB).

    :param n_time: Number of time points.
    :param n_space: Number of spatial points.
    :param innovation_type: Type of innovations, either of 'BM', 'BB'.

    :return: FAR(1) process of shape (n_time, n_space).
    """
    X = np.zeros((n_time, n_space))

    def kernel(tau, sigma):
        return 0.3 * np.sqrt(6) * np.minimum(tau, sigma)

    X[0, :] = np.zeros(n_space)

    epsilon = generate_iid(n_time, n_space, innovation_type)

    for t in range(1, n_time):
        rho_X_prev = apply_rho(X[t - 1], n_space, kernel)
        X[t, :] = rho_X_prev + epsilon[t]

    return X


def generate_hetero_indep(n_time: int, n_space: int) -> np.ndarray:
    """
    Generate a heteroscedastic functional time series of independent processes:
    X_{t,T} = σ(t/T) * B_t, where each B_t is a Brownian motion and
    σ(x) = x + 1/2.

    :param n_time: Number of time points.
    :param n_space: Number of spatial points.
    :return: Functional time series of independent, heteroscedastic processes.
    """
    sigma = np.linspace(0, 1, n_time) + 1 / 2
    bm_time_series = generate_iid(n_time, n_space, 'BM')
    X = sigma[:, np.newaxis] * bm_time_series

    return X


def generate_hetero_dep(n_time: int, n_space: int, model: int = 1) -> np.ndarray:
    """
    Generate a heteroscedastic functional time series of dependent processes:
    Model_1: X_{t,T} = ρ(X_{t-1,T}) + σ(t/T) * B_t,
    Model_2: X_{t,T} = σ(t/T) * ρ(X_{t-1,T}) + B_t
    where B_t is a Brownian motion.

    :param n_time: Number of time points.
    :param n_space: Number of spatial points.
    :param model: Specify which model to use (see above). Either 1 or 2.
    :return: Functional time series of dependent, heteroscedastic processes.
    """
    sigma = np.linspace(0, 1, n_time) + 1 / 2
    bm_time_series = generate_iid(n_time, n_space, 'BM')

    X = np.zeros((n_time, n_space))
    X[0, :] = np.zeros(n_space)

    def kernel(tau, sigma):
        return 0.3 * np.sqrt(6) * np.minimum(tau, sigma)

    for t in range(1, n_time):
        rho_X_prev = apply_rho(X[t - 1, :], n_space, kernel)

        if model == 1:
            X[t, :] = rho_X_prev + sigma[t] * bm_time_series[t]
        elif model == 2:
            X[t, :] = sigma[t] * rho_X_prev + bm_time_series[t]
        else:
            raise ValueError(f"Unknown model {model}.")

    return X
