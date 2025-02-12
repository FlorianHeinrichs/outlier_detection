#
# quantiles.py
#
# Project: Outlier Detection in Time Series
# Date: 2024-09-16
# Author: Florian Heinrichs
#
# Script to estimate quantiles of the maximum over a time series.
# Estimators taken from: https://arxiv.org/pdf/2110.15576
#     Bücher, A., & Zanger, L. (2023). On the disjoint and sliding block maxima
#     method for piecewise stationary time series. The Annals of Statistics,
#     51(2), 573-598.
# See (10) and subsequent equations


import numpy as np
from scipy.optimize import newton
from scipy.special import gamma as gamma_func


def get_block_maxima(X: np.ndarray, block_size: int = None,
                     method: str = 'sliding') -> np.ndarray:
    """
    Get sliding or disjoint block maxima of a given dataset.

    :param X: One-dimensional data as NumPy array of size (n_timeseries, n) or
        (n,).
    :param block_size: Number of elements to consider for each block maximum.
    :param method: String indicating 'sliding' or 'disjoint' block maxima.
    :return: Returns NumPy array of size
    """
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
        squeeze = True
    elif len(X.shape) == 2:
        squeeze = False
    else:
        raise ValueError(f'Shape of X is not supported: {X.shape=}.')

    n = X.shape[-1]

    if block_size is None:
        block_size = np.ceil(np.sqrt(n)).astype('int')

    if method == 'sliding':
        windows = np.lib.stride_tricks.sliding_window_view(
            X, window_shape=block_size, axis=1)
        block_max = np.max(windows, axis=-1)
    elif method == 'disjoint':
        n_blocks = n // block_size
        new_shape = (X.shape[0], n_blocks, block_size)
        X = X[:, :n_blocks * block_size].reshape(new_shape)
        block_max = np.max(X, axis=-1)
    else:
        raise ValueError('Method must be either sliding or disjoint.')

    if squeeze:
        block_max = np.squeeze(block_max, axis=0)

    return block_max


def get_quantile(X: np.ndarray, alpha: float | np.ndarray,
                 block_size: int = None, method: str = 'sliding') -> np.float64:
    """
    Estimate 1-alpha quantile(s) based on a given dataset.

    :param X: One-dimensional data as NumPy array of size (n_timeseries, n) or
        (n,).
    :param alpha: Level of quantile to estimate, given as float or NumPy array.
    :param block_size: Number of elements to consider for each block maximum.
    :param method: String indicating 'sliding' or 'disjoint' block maxima.
    :return: Estimated 1-alpha quantile(s).
    """
    n = X.shape[-1]

    if block_size is None:
        block_size = np.ceil(np.sqrt(n)).astype('int')

    block_maxima = get_block_maxima(X=X, block_size=block_size, method=method)
    gamma, sigma_r, mu_r = get_estimators(block_maxima)

    block_maxima_2 = get_block_maxima(X=X, block_size=2 * block_size,
                                      method=method)
    _, _, mu_2r = get_estimators(block_maxima_2)

    coeff = np.ones_like(gamma) * np.log(n / block_size) / np.log(2)
    ind = np.abs(gamma) > 1e-10
    coeff[ind] = ((n / block_size) ** gamma[ind] - 1) / (2 ** gamma[ind] - 1)

    sigma = sigma_r * (n / block_size) ** gamma
    mu = mu_r + (mu_2r - mu_r) * coeff

    if isinstance(alpha, float):
        alpha = np.array([[alpha]])
    elif isinstance(alpha, np.ndarray) and len(alpha.shape) == 1:
        alpha = np.expand_dims(alpha, axis=0)

    mu = np.expand_dims(mu, axis=-1)
    sigma = np.expand_dims(sigma, axis=-1)
    gamma = np.expand_dims(gamma, axis=-1)

    quantile = mu - sigma * np.log(- np.log(1 - alpha))
    quantile[ind] = mu[ind] + sigma[ind] / gamma[ind] * (
            (- np.log(1 - alpha)) ** (-gamma[ind]) - 1)


    return quantile


def get_estimators(M: np.ndarray) -> tuple:
    """
    Calculate estimators beta_0, beta_1, beta_2 based on block maxima M.

    :param M: Block maxima as NumPy array.
    :return: Estimators as NumPy array.
    """
    beta_0 = get_beta_0(M)
    beta_1 = get_beta_1(M)
    beta_2 = get_beta_2(M)

    gamma, sigma, mu = solve_equations(beta_0, beta_1, beta_2)

    return gamma, sigma, mu


def get_beta_0(M: np.ndarray) -> np.float64:
    """
    Calculate estimator of beta_0 based on block maxima M.

    :param M: Block maxima as NumPy array.
    :return: Estimator of beta_0.
    """
    return np.mean(M, axis=-1)


def get_beta_1(M: np.ndarray) -> np.float64:
    """
    Calculate estimator of beta_1 based on block maxima M.

    :param M: Block maxima as NumPy array.
    :return: Estimator of beta_1.
    """
    n = M.shape[-1]

    if n < 2:
        raise ValueError('Block maxima must have at least two elements.')

    M = np.sort(M, axis=-1)
    weights = np.arange(n) / (n - 1)

    if len(M.shape) == 2:
        weights = np.expand_dims(weights, axis=0)

    return np.mean(weights * M, axis=-1)


def get_beta_2(M: np.ndarray) -> np.float64:
    """
    Calculate estimator of beta_2 based on block maxima M.

    :param M: Block maxima as NumPy array.
    :return: Estimator of beta_2.
    """
    n = M.shape[-1]

    if n < 3:
        raise ValueError('Block maxima must have at least three elements.')

    M = np.sort(M, axis=-1)
    weights = np.arange(n) * np.arange(-1, n - 1) / ((n - 1) * (n - 2))

    if len(M.shape) == 2:
        weights = np.expand_dims(weights, axis=0)

    return np.mean(weights * M, axis=-1)


def solve_equations(beta_0: np.ndarray, beta_1: np.ndarray,
                    beta_2: np.ndarray) -> tuple:
    arg1 = (3 * beta_2 - beta_0) / (2 * beta_1 - beta_0)

    gamma = np.zeros_like(arg1)
    ind = np.abs(arg1 - np.log(3) / np.log(2)) > 1e-5
    gamma[ind] = newton(g1, np.ones(np.sum(ind)), fprime=g1_prime,
                        args=(arg1[ind],))

    sigma = g2(gamma) * (2 * beta_1 - beta_0)
    mu = beta_0 + sigma * g3(gamma)

    return gamma, sigma, mu

def g1(x, y):
    return (3 ** x - 1) / (2 ** x - 1) - y


def g1_prime(x, y):
    return ((np.log(3) * 3 ** x * (2 ** x - 1)
            - np.log(2) * 2 ** x * (3 ** x - 1)) / (2 ** x - 1) ** 2)

def g2(x):
    g2_value = np.ones_like(x) / np.log(2)
    ind = np.abs(x) > 1e-5
    g2_value[ind] = x[ind] / (gamma_func(1 - x[ind]) * (2 ** x[ind] - 1))

    return g2_value

def g3(x):
    g3_value = - np.euler_gamma * np.ones_like(x)
    ind = np.abs(x) > 1e-5
    g3_value[ind] = (1 - gamma_func(1 - x[ind])) / x[ind]

    return g3_value
