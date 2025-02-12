#
# experiments_config.py
#
# Project: Outlier Detection in Time Series
# Date: 2024-11-22
# Author: Florian Heinrichs
#
# Config file containing configuration settings for experiments.


import numpy as np

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

OUTLIER_RATE = 0.05
ALPHA = 0.01

CV_KWARGS = {'num_folds': 5, 'min_bw': 2, 'max_bw': 20}

def get_config(outlier: bool) -> list:
    """
    Get list of full experiment config.

    :param outlier: Indicator, whether to use outliers.
    :return: List with full experiment config.
    """
    config = []

    for m_stable in M_STABLE:
        n = N_FACTOR * m_stable
        data_config = get_data_config(n, N_SAMPLES)
        quantile_kwargs = {'alpha': ALPHA * np.ones(n - m_stable)}

        for data_kwargs in data_config:
            error_kwargs = data_kwargs['error_kwargs']

            if 'distribution' in error_kwargs:
                error_dist = error_kwargs['distribution']
            else:
                error_dist = error_kwargs['error_dist']

            outlier_height = get_minimal_height(m_stable, error_dist)
            outlier_kwargs = {'min_height': outlier_height,
                              'n_outliers': int((n - m_stable) * OUTLIER_RATE)}

            args = data_kwargs, m_stable, CV_KWARGS, quantile_kwargs
            kwargs = {'outlier_kwargs': outlier_kwargs if outlier else None}
            experiment_id = get_experiment_id(args, kwargs)
            config.append((experiment_id, args, kwargs))

    return config

def get_minimal_height(n: int, distribution: str) -> float:
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
    :return: Minimal height.
    """

    if distribution == 'normal':
        c_n = np.sqrt(2 * np.log(n))
    elif distribution == 'uniform':
        c_n = 1 - 1/n
    elif distribution == 'exponential':
        c_n = np.log(n)
    elif distribution == 'pareto':
        c_n = n ** (1 / 4)
    elif distribution == 'pareto_infinite_variance':
        c_n = np.sqrt(n)
    else:
        raise ValueError(f"{distribution=} unknown.")

    # Factor originally so that minimal height was log(n)/10 for normal dist.
    factor = np.sqrt(np.log(n) / 200)

    return factor * c_n


def get_experiment_id(args: tuple, kwargs: dict) -> str:
    data_kwargs = args[0]
    m_stable = args[1]
    outlier = 'no_outlier' if kwargs['outlier_kwargs'] is None else 'outlier'

    mean_type = data_kwargs['mean_type']
    error_type = data_kwargs['error_type']
    error_kwargs = data_kwargs['error_kwargs']

    if 'distribution' in error_kwargs:
        error_dist = error_kwargs['distribution']
    else:
        error_dist = error_kwargs['error_dist']

    experiment_id = (f"{outlier}_{mean_type}_{error_type}_{error_dist}_"
                     f"{m_stable}")

    return experiment_id


def get_data_config(n: int, n_samples: int) -> list:
    """
    Get list of all data config combinations.

    :param n: Length of time series.
    :param n_samples: Number of (independent) time series.
    :return: List with all config combinations.
    """
    data_config = []

    for mean_type, mean_kwargs in MEAN_CONFIG:
        mean_kwargs = mean_kwargs.copy()
        mean_kwargs['n'] = n

        for error_type in ERROR_TYPES:
            for error_dist in ERROR_DISTRIBUTIONS:
                error_kwargs = {'n': n, 'n_samples': n_samples, 'sigma': SIGMA}

                if error_type == 'iid':
                    error_kwargs['distribution'] = error_dist
                else:
                    error_kwargs['error_dist'] = error_dist

                data_kwargs = {'mean_type': mean_type,
                               'mean_kwargs': mean_kwargs,
                               'error_type': error_type,
                               'error_kwargs': error_kwargs}

                data_config.append(data_kwargs)

    return data_config


if __name__ == '__main__':
    for error in ERROR_DISTRIBUTIONS:
        for n in M_STABLE:
            print(f"{error}, {n}, {get_minimal_height(n, error)}")
