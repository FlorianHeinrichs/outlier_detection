#
# experiments_config.py
#
# Project: Outlier Detection in Time Series
# Date: 2024-11-22
# Author: Florian Heinrichs
#
# Config file containing configuration settings for experiments.

# Mean config containing tuple of mean_type and mean_kwargs

import numpy as np

from experiments_config import get_data_config

MEAN_CONFIG = [('1', {'a': 1}),
               ('2', {}),
               ('abrupt', {'cp': 0.5}),
               ('const', {})]

SIGMA = 1 / 20

ERROR_TYPES = ['iid', 'ar', 'ma']
ERROR_DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 'pareto',
                       'pareto_infinite_variance']

M_STABLE = [50, 100, 200]
N_FACTOR = 11
N_SAMPLES = 1000
# N_SAMPLES = 10

OUTLIER_RATE = 0.05
# ALPHA = [0.01, 0.05, 0.1]
ALPHA_FIX = 0.05
ALPHA = [ALPHA_FIX]

HEIGHT_FACTORS = [1/8, 1/4, 1/2, 1, 2]

CV_KWARGS = {'num_folds': 5, 'min_bw': 2, 'max_bw': 20}


def get_minimal_height(n: int, distribution: str, std: float) -> float:
    """
    Calculates minimal outlier height so that the test is consistent.

    :param n: Length of stable period.
    :param distribution: Error distribution.
        - The dominating sequences are:
        - 'normal': a_n= \sqrt{2 \log(n)}
        - 'uniform': b_n = 1 - 1/n
        - 'exponential': b_n = \log(n) / \lambda
        - 'pareto'/'pareto_infinite_variance': a_n = n^{1/a}
        - See: http://thierry-roncalli.com/download/HFRM-Chap12.pdf Table 12.4
        and Example 127
    :param std: Standard deviation of errors.
    :return: Minimal height.
    """
    if distribution == 'normal':
        bn = (np.sqrt(2 * np.log(2 * n))
              - (np.log(np.log(2 * n)) + np.log(4 * np.pi))
              / (2 * np.sqrt(2 * np.log(2 * n))))
        an = 1 / np.sqrt(2 * np.log(2 * n))
    elif distribution == 'uniform':
        bn =  np.sqrt(3)
        an = np.sqrt(12) / (2 * n)
    elif distribution == 'exponential':
        bn = np.log(n) - 1
        an = 1
    elif distribution == 'pareto':
        bn = 3 / np.sqrt(2) * n ** (1 / 4)
        an = bn / 4
    elif distribution == 'pareto_infinite_variance':
        bn = np.sqrt(n)
        an = bn / 2
    else:
        raise ValueError(f"{distribution=} unknown.")

    bn, an = bn * std, an * std
    factor = np.log(n) ** 2
    cn = factor * an + bn

    return cn


def get_config_general(outlier: bool) -> list:
    """
    Get list of full experiment config.

    :param outlier: Indicator, whether to use outliers.
    :return: List with full experiment config.
    """
    config = []

    for alpha in ALPHA:
        for m_stable in M_STABLE:
            n = N_FACTOR * m_stable
            data_config = get_data_config(n, N_SAMPLES)
            quantile_kwargs = {'alpha': alpha * np.ones(n - m_stable),
                               'return_estimates': True}

            for data_kwargs in data_config:
                error_kwargs = data_kwargs['error_kwargs']
                std = error_kwargs['sigma']

                if 'distribution' in error_kwargs:
                    error_dist = error_kwargs['distribution']
                else:
                    error_dist = error_kwargs['error_dist']

                outlier_height = get_minimal_height(m_stable, error_dist, std=std)
                outlier_kwargs = {
                    'min_height': outlier_height,
                    'n_outliers': int((n - m_stable) * OUTLIER_RATE)
                }

                args = data_kwargs, m_stable, CV_KWARGS, quantile_kwargs
                kwargs = {'outlier_kwargs': outlier_kwargs if outlier else None}
                experiment_id = get_experiment_id(args, kwargs)
                config.append((experiment_id, args, kwargs))

    return config


def get_config_outlier_height():
    """
    Get list of experiment config for tests on robustness against different
    outlier heights.

    :return: List with experiment config.
    """
    config = []

    alpha = 0.05

    for height_factor in HEIGHT_FACTORS:

        for m_stable in M_STABLE:
            n = N_FACTOR * m_stable

            error_dist = 'normal'
            error_kwargs = {'n': n, 'n_samples': N_SAMPLES,
                            'sigma': SIGMA, 'distribution': error_dist}

            data_kwargs = {'mean_type': 'const',
                           'mean_kwargs': {'n': n},
                           'error_type': 'iid',
                           'error_kwargs': error_kwargs}

            quantile_kwargs = {'alpha': alpha * np.ones(n - m_stable),
                               'return_estimates': True}

            outlier_height = get_minimal_height(m_stable, error_dist, std=SIGMA)
            outlier_kwargs = {
                'min_height': height_factor * outlier_height,
                'n_outliers': int((n - m_stable) * OUTLIER_RATE)
            }

            args = data_kwargs, m_stable, CV_KWARGS, quantile_kwargs
            kwargs = {'outlier_kwargs': outlier_kwargs}
            experiment_id = f"{m_stable}_height{height_factor}"
            config.append((experiment_id, args, kwargs))

    return config


def get_config_outlier_stable():
    """
    Get list of experiment config for tests on robustness against outliers in
    initial "stable" period, that is outlier free by assumption.

    :return: List with experiment config.
    """
    config = []

    alpha = 0.05

    for outlier_rate in [0.02, 0.04, 0.06, 0.08, 0.1]:

        for m_stable in M_STABLE:
            n = N_FACTOR * m_stable

            error_dist = 'normal'
            error_kwargs = {'n': n, 'n_samples': N_SAMPLES,
                            'sigma': SIGMA, 'distribution': error_dist}

            data_kwargs = {'mean_type': 'const',
                           'mean_kwargs': {'n': n},
                           'error_type': 'iid',
                           'error_kwargs': error_kwargs}

            quantile_kwargs = {'alpha': alpha * np.ones(n - m_stable),
                               'return_estimates': True}

            outlier_height = get_minimal_height(m_stable, error_dist, std=SIGMA)
            outlier_kwargs = {
                'min_height': outlier_height,
                'n_outliers': int((n - m_stable) * OUTLIER_RATE),
                'n_outliers_stable': int(m_stable * outlier_rate)
            }

            args = data_kwargs, m_stable, CV_KWARGS, quantile_kwargs
            kwargs = {'outlier_kwargs': outlier_kwargs}
            experiment_id = f"{m_stable}_rate{outlier_rate}"
            config.append((experiment_id, args, kwargs))

    return config


def get_config_short(outlier: bool) -> list:
    """
    Get list of full experiment config (N_FACTOR = 2).

    :param outlier: Indicator, whether to use outliers.
    :return: List with full experiment config.
    """
    config = []

    for m_stable in M_STABLE:
        n = 2 * m_stable
        data_config = get_data_config(n, N_SAMPLES)
        quantile_kwargs = {'alpha': ALPHA_FIX * np.ones(n - m_stable),
                           'return_estimates': True}

        for data_kwargs in data_config:
            error_kwargs = data_kwargs['error_kwargs']
            std = error_kwargs['sigma']

            if 'distribution' in error_kwargs:
                error_dist = error_kwargs['distribution']
            else:
                error_dist = error_kwargs['error_dist']

            outlier_height = get_minimal_height(m_stable, error_dist, std=std)
            outlier_kwargs = {
                'min_height': outlier_height,
                'n_outliers': int((n - m_stable) * OUTLIER_RATE)
            }

            args = data_kwargs, m_stable, CV_KWARGS, quantile_kwargs
            kwargs = {'outlier_kwargs': outlier_kwargs if outlier else None}
            experiment_id = get_experiment_id(args, kwargs)
            experiment_id = experiment_id + f"_n{n}"
            config.append((experiment_id, args, kwargs))

    return config


def get_experiment_id(args: tuple, kwargs: dict) -> str:
    data_kwargs = args[0]
    m_stable = args[1]
    quantile_kwargs = args[3]
    outlier = 'no_outlier' if kwargs['outlier_kwargs'] is None else 'outlier'

    mean_type = data_kwargs['mean_type']
    error_type = data_kwargs['error_type']
    error_kwargs = data_kwargs['error_kwargs']
    alpha = int(100 * quantile_kwargs['alpha'][0])

    if 'distribution' in error_kwargs:
        error_dist = error_kwargs['distribution']
    else:
        error_dist = error_kwargs['error_dist']

    experiment_id = (f"{outlier}_{mean_type}_{error_type}_{error_dist}_"
                     f"{m_stable}_{alpha}")

    return experiment_id


if __name__ == '__main__':
    # for error in ERROR_DISTRIBUTIONS:
    #     for n in M_STABLE:
    #         print(f"{error}, {n}, {get_minimal_height(n, error)}")

    import matplotlib.pyplot as plt
    from tueplots import bundles, figsizes
    from data_generation import mu_1, mu_2, mu_abrupt, generate_iid

    plt.rcParams.update(bundles.icml2022(family="Times New Roman", usetex=False))
    plt.rcParams.update(figsizes.icml2022_full())

    m = 50
    n = 11
    x1 = mu_1(n * m, 1)
    x2 = mu_2(n * m)
    x3 = mu_abrupt(n * m, 0.5)
    xs = [x1, x2, x3]

    distributions = ['uniform', 'uniform']
    fig, axes = plt.subplots(1, 2)

    for i, (ax, x, dist) in enumerate(zip(axes, xs, distributions)):
        eps = generate_iid(n * m, 1, distribution=dist, sigma=SIGMA)

        min_height = get_minimal_height(m, dist, SIGMA)
        n_outliers = int((n - 1) * m * OUTLIER_RATE)
        outliers_index = np.random.choice(
            np.arange((n - 1) * m), size=n_outliers, replace=False
        ) + m
        outlier_heights = min_height * np.random.uniform(
            1, 2, size=(n_outliers,))
        outlier_sign = 2 * np.random.randint(0, 2, (n_outliers,)) - 1
        outliers = outlier_sign * outlier_heights
        signal = x + eps
        signal[:, outliers_index] = signal[:, outliers_index] + outliers


        ax.plot(np.linspace(0, 11, n * m), signal[0], zorder=1)
        ax.scatter(outliers_index / m, signal[:, outliers_index], c='tab:orange', zorder=2)

        # ax.set_ylim([0.48, 1.08])
        ax.set_title(label=f"$\\nu_{i+1}$")

    plt.show()
