#
# data_generation.py
#
# Project: Outlier Detection in Time Series
# Date: 2024-09-18
# Author: Florian Heinrichs
#
# Script to generate data, data generating processes based on the paper:
# - "Bücher, A., Dette, H., & Heinrichs, F. (2021). Are deviations in a
# gradually varying mean relevant? A testing approach based on sup-norm
# estimators. The Annals of Statistics, 49(6), 3583-3617."

import numpy as np


def generate_data(mean_type: str, mean_kwargs: dict,
                  error_type: str, error_kwargs: dict) -> np.ndarray:
    """
    Wrapper function to generate noisy data.

    :param mean_type: Type of mean in ['1', '2', 'abrupt', 'const']
    :param mean_kwargs: Dictionary containing settings for mean.
    :param error_type: Type of error in ['iid', 'ar', 'ma']
    :param error_kwargs: Dictionary containing settings for errors.
    :return: NumPy array containing noisy data.
    """
    if mean_type == '1':
        mean = mu_1(**mean_kwargs)
    elif mean_type == '2':
        mean = mu_2(**mean_kwargs)
    elif mean_type == 'abrupt':
        mean = mu_abrupt(**mean_kwargs)
    elif mean_type == 'const':
        mean = np.ones(mean_kwargs.get('n', 1))
    else:
        raise ValueError("Mean type unknown.")

    if error_type == 'iid':
        error = generate_iid(**error_kwargs)
    elif error_type == 'ar':
        error = generate_ar(**error_kwargs)
    elif error_type == 'ma':
        error = generate_ma(**error_kwargs)
    else:
        raise ValueError("Error type unknown.")

    return mean + error


def mu_1(n: int, a: float = 1.) -> np.ndarray:
    """
    Generates a non-monotonically decreasing function (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :param a: Scaling parameter for quadratic term.
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    mu = a * (x - 1 / 2) ** 2 + np.sin(5 * 2 * np.pi * x) / 10 + 3/4

    return mu.reshape((1, n))


def mu_2(n: int) -> np.ndarray:
    """
    Generates a monotonically decreasing function (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    mu = - np.sin(2 * np.pi * np.linspace(0, 1, n))
    mu[:n // 4] = -1
    mu[(3 * n) // 4:] = 1
    mu = mu / 4 + 0.75

    return mu.reshape((1, n))


def mu_abrupt(n: int, cp: float) -> np.ndarray:
    """
    Generates a step function with change point at "cp" (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :param cp: Position of change point in the unit interval (as float).
    :return: Function values at points np.arange(n) / n
    """
    mu = np.ones((1, n))
    mu[:, :int(cp * n)] = 1/2

    return mu


def generate_iid(n: int, n_samples: int, distribution: str = 'normal',
                 mean: float = 0., sigma: float = 1.) -> np.ndarray:
    """
    Generates i.i.d. errors according to the specified distribution.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param distribution: Distribution to be used.
    :param mean: Mean of errors.
    :param sigma: Standard deviation of errors.
    :return: Errors at points np.arange(n) / n. Output has shape (n_samples, n).
    """
    rng = np.random.default_rng()

    if distribution == 'normal':
        errors = rng.normal(loc=0, scale=1, size=(n_samples, n))

    elif distribution == 'uniform':
        errors = rng.uniform(low=0, high=1, size=(n_samples, n))
        errors = np.sqrt(12) * (errors - 1/2)

    elif distribution == 'exponential':
        errors = rng.exponential(scale=1, size=(n_samples, n)) - 1

    elif distribution == 'pareto':
        errors = rng.pareto(4, size=(n_samples, n))
        errors = np.sqrt(9/2) * (errors - 1/3)

    elif distribution == 'pareto_infinite_variance':
        errors = rng.pareto(2, size=(n_samples, n)) - 1

    else:
        raise ValueError("Distribution unknown.")

    errors = errors * sigma + mean

    return errors


def generate_ar(n: int, n_samples: int, sigma: float = 1., burn_in: int = 100,
                a: float = 0.5, error_dist: str = 'normal') -> np.ndarray:
    """
    Generates AR(1) errors.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param sigma: Standard deviation of errors.
    :param burn_in: Time steps used for burn in of AR process.
    :param a: Autoregressive coefficient.
    :param error_dist: Distribution of errors.
    :return: Errors at points np.arange(n) / n. Output has shape (n_samples, n).
    """
    epsilon = generate_iid(n + burn_in, n_samples, mean=0., sigma=1.,
                           distribution=error_dist)

    errors = np.zeros((n_samples, n + burn_in))

    errors[:, 0] = epsilon[:, 0]
    for i in range(1, n + burn_in):
        errors[:, i] = a * errors[:, i - 1] + epsilon[:, i]

    var_errors = 1 / (1 - a ** 2)
    errors = sigma / np.sqrt(var_errors) * errors[:, burn_in:]

    return errors


def generate_ma(n: int, n_samples: int, sigma: float = 1.,
                a: float = 0.5, error_dist: str = 'normal') -> np.ndarray:
    """
    Generates MA(1) errors.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param sigma: Standard deviation of errors.
    :param a: MA coefficient.
    :param error_dist: Distribution of errors.
    :return: Errors at points np.arange(n) / n. Output has shape (n_samples, n).
    """
    epsilon = generate_iid(n + 1, n_samples, mean=0., sigma=1.,
                           distribution=error_dist)

    errors = epsilon[:, 1:] + a * epsilon[:, :-1]

    var_errors = 1 + a ** 2
    errors = sigma / np.sqrt(var_errors) * errors

    return errors
