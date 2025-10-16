#
# experiments.py
#
# Project: Outlier Detection in Time Series
# Date: 2024-09-18
# Author: Florian Heinrichs
#
# Main script containing experiments.

import csv
from datetime import datetime
from typing import Callable

import numpy as np

from alternative_methods import campulova_2018, holesovsky_2018, ml_based
from data_generation import generate_data
from experiments_config_v2 import (get_config_general,
                                   get_config_outlier_height,
                                   get_config_outlier_stable,
                                   get_config_short)
from jackknife import bandwidth_cv, jackknife_estimation
from quantiles import get_quantile, get_true_quantile


def parallel_test(data: np.ndarray,
                  bw: int | np.ndarray,
                  m_stable: int,
                  quantile_kwargs: dict = None,
                  quantile: np.ndarray | float = None,
                  estimation: Callable = None) -> np.ndarray | tuple:
    """
    Auxiliary function to conduct parallel tests.

    :param data: NumPy array containing the data, of size
        (n_time_series, n_samples_per_ts).
    :param bw: Bandwidth to use for kernel regression (can vary per time series).
    :param m_stable: Length of stable period without outliers.
    :param quantile_kwargs: Dictionary containing keyword arguments for
        quantiles.
    :param quantile: Quantile to use for test. If not specified, it is estimated.
    :param estimation: Function used to smooth the data. Defaults to
        jackknife estimation.
    :return: NumPy array containing test decisions. If 'return_estimates' in
        the quantile_kwargs is True, additional estimates of parameters are
        returned.
    """
    if quantile_kwargs is None:
        quantile_kwargs = {}

    if estimation is None:
        estimation = jackknife_estimation

    mean_estimator = estimation(data, bw)
    residuals = data - mean_estimator
    residuals_stable = residuals[..., :m_stable]

    return_estimates = quantile_kwargs.get('return_estimates', False)

    if quantile is None:
        if return_estimates:
            quantile, mu, sigma, gamma = get_quantile(residuals_stable,
                                                      **quantile_kwargs)
        else:
            quantile = get_quantile(residuals_stable, **quantile_kwargs)

    residuals = residuals[..., m_stable:]
    reject_null = np.abs(residuals) > quantile

    if return_estimates:
        return reject_null, mu, sigma, gamma
    else:
        return reject_null

def sequential_test(data: np.ndarray,
                    bw: int | np.ndarray,
                    m_stable: int,
                    m_remaining: int,
                    quantile_kwargs: dict = None,
                    quantile: np.ndarray | float = None,
                    estimation: Callable = None) -> np.ndarray | tuple:
    """
    Auxiliary function to conduct sequential tests.

    :param data: NumPy array containing the data, of size
        (n_time_series, n_samples_per_ts).
    :param bw: Bandwidth to use for kernel regression (can vary per time series).
    :param m_stable: Length of stable period without outliers.
    :param m_remaining: Length of remaining time series, should coincide with
        n_samples_per_ts - m_stable.
    :param quantile_kwargs: Dictionary containing keyword arguments for
        quantiles.
    :param quantile: Quantile to use for test. If not specified, it is estimated.
    :param estimation: Function used to smooth the data. Defaults to
        jackknife estimation.
    :return: NumPy array containing test decisions. If 'return_estimates' in
        the quantile_kwargs is True, additional estimates of parameters are
        returned.
    """
    n_time_series = data.shape[0]

    if quantile_kwargs is None:
        quantile_kwargs = {}

    if estimation is None:
        estimation = jackknife_estimation

    mean_estimator = estimation(data[..., :m_stable], bw)
    residuals_stable = data[..., :m_stable] - mean_estimator

    return_estimates = quantile_kwargs.get('return_estimates', False)

    if quantile is None:
        if return_estimates:
            quantile, mu, sigma, gamma = get_quantile(residuals_stable,
                                                      **quantile_kwargs)
        else:
            quantile = get_quantile(residuals_stable, **quantile_kwargs)
    elif isinstance(quantile, np.ndarray) and len(quantile.shape) == 1:
        quantile = np.expand_dims(quantile, axis=0)

    reject_null = np.zeros((n_time_series, m_remaining), dtype=bool)

    for time_step in range(m_remaining):
        time_idx = time_step + m_stable
        max_bw = np.max(bw)
        current_data = data[..., time_idx - max_bw: time_idx + 1]

        row_indices = np.expand_dims(np.arange(n_time_series), axis=-1)
        col_indices = np.arange(max_bw + 1)
        mask = col_indices < (max_bw - bw)[row_indices]

        outlier_mask = np.zeros((n_time_series, max_bw), dtype=bool)
        if 0 < time_step < max_bw:
            outlier_mask[:, -time_step:] = reject_null[:, :time_step]
        elif max_bw <= time_step:
            outlier_mask = reject_null[:, time_step - max_bw: time_step]

        mask[:, :-1] = mask[:, :-1] | outlier_mask

        mean_estimator = estimation(current_data, bw, mask=mask)
        residuals = (current_data - mean_estimator)[:, -1]

        reject_null_t = np.abs(residuals) > quantile[:, time_step]
        reject_null[:, time_step] = reject_null_t

    if return_estimates:
        return reject_null, mu, sigma, gamma
    else:
        return reject_null

def alternative_tests(data: np.ndarray, m_stable: int, m_remaining: int,
                      alpha: float, outliers_index: np.ndarray = None,
                      return_rejections: bool = False,
                      methods: list = None) -> dict:
    """
    Auxiliary function to conduct alternative tests.

    :param data: NumPy array containing the data, of size
        (n_time_series, n_samples_per_ts).
    :param m_stable: Length of stable period without outliers.
    :param m_remaining: Length of remaining time series, should coincide with
        n_samples_per_ts - m_stable.
    :param alpha: Test level for m_stable many hypothesis
    :param outliers_index: NumPy array of outlier indices.
    :param return_rejections: Boolean, indicating if the individual test
        decisions should be returned.
    :param methods: List of alternative methods that should be used.
    :return: Dictionary containing test results.
    """
    alternatives = {'Campulova2018': campulova_2018,
                    'Holesovsky2018': holesovsky_2018}

    if methods is None:
        methods = ['Campulova2018','Holesovsky2018', 'Wette2024',
                   'Malhotra2015', 'Munir2018']

    results = {}

    for method in methods:
        func = alternatives.get(method)

        if func is None:
            rej_null = ml_based(data[:, :m_stable], data[:, m_stable:],
                                alpha=alpha, n=m_stable, od_method=method,
                                method='chebyshev')
        else:
            rej_null = func(data, alpha=alpha, n=m_stable)[:, m_stable:]

        emp_rej_rate = np.sum(rej_null, axis=-1) / m_remaining

        if outliers_index is None:
            results[method] = emp_rej_rate
        else:
            cm = calculate_confusion_matrix(outliers_index - m_stable, rej_null)
            results[method] = emp_rej_rate, cm

        if return_rejections:
            results[method] = *results[method], rej_null

    return results

def experiment(data_kwargs: dict,
               m_stable: int,
               cv_kwargs: dict,
               quantile_kwargs: dict,
               outlier_kwargs: dict = None,
               methods: list = None,
               modes: list = None) -> dict:
    """
    Main function for running the experiments.

    :param data_kwargs: Dictionary containing settings for data generation.
    :param m_stable: Length of stable period without outliers.
    :param cv_kwargs: Dictionary containing settings for cross-validation to
        optimize the jackknife estimator's bandwidth.
    :param quantile_kwargs: Dictionary containing settings for quantile
        estimation.
    :param outlier_kwargs: Dictionary containing settings for outlier generation.
    :param methods: List of alternative methods that should be used.
    :param modes: Modes of the proposed testing procedure. Valid values:
        - 'parallel'
        - 'sequential'
        - 'parallel (true quantile)'
        - 'sequential (true quantile)'
    :return: Empirical rejection rate as float and confusion matrix as NumPy
        array.
    """
    if modes is None:
        modes = ['parallel', 'sequential']

    if methods is None:
        methods = []

    # Generate Data
    data = generate_data(**data_kwargs)
    bw = bandwidth_cv(data[..., :m_stable], **cv_kwargs)
    m_remaining = data.shape[-1] - m_stable
    n_time_series = data.shape[0]

    # Generate Outliers
    if outlier_kwargs:
        min_height = outlier_kwargs.get('min_height', 0)
        n_outliers = outlier_kwargs.get('n_outliers', 0)
        n_outliers_stable = outlier_kwargs.get('n_outliers_stable', 0)
        outliers_index = np.stack([
            np.random.choice(
                np.arange(m_remaining), size=n_outliers, replace=False
            ) + m_stable for _ in range(n_time_series)
        ])
        outlier_heights = min_height * np.random.uniform(
            1, 2, size=(n_time_series, n_outliers))
        outlier_sign = 2 * np.random.randint(0, 2, (n_time_series, n_outliers)) - 1
        outliers = outlier_sign * outlier_heights
        index = np.expand_dims(np.arange(n_time_series), axis=-1)
        data[index, outliers_index] = data[index, outliers_index] + outliers

        if n_outliers_stable > 0:
            outliers_stable = np.stack([
                np.random.choice(
                    np.arange(m_stable), size=n_outliers_stable, replace=False
                ) for _ in range(n_time_series)
            ])
            outlier_heights = min_height * np.random.uniform(
                1, 2, size=(n_time_series, n_outliers_stable))
            outlier_sign = 2 * np.random.randint(
                0, 2, (n_time_series, n_outliers_stable)) - 1
            outliers = outlier_sign * outlier_heights
            index = np.expand_dims(np.arange(n_time_series), axis=-1)
            data[index, outliers_stable] = data[index, outliers_stable] + outliers

    else:
        outliers_index = np.zeros(0, dtype=np.int32)

    results = {}

    for mode in modes:
        if 'true quantile' in mode:
            alpha = quantile_kwargs.get('alpha', 0.05)
            error_kwargs = data_kwargs['error_kwargs']
            distribution = error_kwargs.get('distribution',
                                            error_kwargs.get('error_dist'))
            std = error_kwargs.get('sigma')
            quantile, mu, sigma, gamma = get_true_quantile(
                alpha, distribution, m_stable, return_params=True, std=std)

            if mode == 'parallel (true quantile)':
                reject_null = parallel_test(data, bw, m_stable,
                                            quantile=quantile)
            elif mode == 'sequential (true quantile)':
                reject_null = sequential_test(data, bw, m_stable, m_remaining,
                                              quantile=quantile)
            else:
                raise ValueError(f"Mode {mode} unknown.")
        elif mode == 'parallel':
            result = parallel_test(data, bw, m_stable, quantile_kwargs)
            reject_null, mu, sigma, gamma = result
        elif mode == 'sequential':
            result = sequential_test(data, bw, m_stable, m_remaining,
                                     quantile_kwargs)
            reject_null, mu, sigma, gamma = result
        else:
            raise ValueError(f"Mode {mode} unknown.")

        empirical_rejection_rate = np.sum(reject_null, axis=-1) / m_remaining

        confusion_matrix = calculate_confusion_matrix(
            outliers_index - m_stable, reject_null)

        results[f"Ours ({mode})"] = empirical_rejection_rate, confusion_matrix
        results[f"Estimates ({mode})"] = (np.mean(mu), np.std(mu),
                                          np.mean(sigma), np.std(sigma),
                                          np.mean(gamma), np.std(gamma))

    alpha = float(quantile_kwargs['alpha'][0])
    alternatives = alternative_tests(data, m_stable, m_remaining, alpha,
                                     outliers_index, methods=methods)
    results.update(alternatives)

    return results


def calculate_confusion_matrix(real_outliers, test_decision) -> np.ndarray:
    """
    Calculate confusion matrix based on detected and real outliers.

    :param real_outliers: NumPy array containing indices of real outliers.
    :param test_decision: NumPy array containing the test decision (0 or 1) of
        size (n_time_series, m_remaining).
    :return: Confusion matrix based on detected and real outliers.
    """
    real = np.zeros_like(test_decision, dtype=bool)
    index = np.expand_dims(np.arange(len(test_decision)), axis=-1)
    real[index, real_outliers] = True

    tp = np.sum(test_decision * real, axis=-1)
    tn = np.sum((1 - test_decision) * (1 - real), axis=-1)
    fn = np.sum(real * (1 - test_decision), axis=-1)
    fp = np.sum((1 - real) * test_decision, axis=-1)

    confusion_matrix = np.array([[tp, fn],
                                 [fp, tn]])
    confusion_matrix = np.transpose(confusion_matrix, axes=[2, 0, 1])

    return confusion_matrix


def prepare_logfiles(methods: list) -> tuple:
    """
    Auxiliary function to prepare log files for results:
    - per_point: statistics are based on individual observations, e.g. the
        number of false positives (FPs) coincides with the average FPs over
        all time series.
    - per_time_series: statistics are based on aggregated time series, e.g. the
        number of false positives (FPs) coincides with the number of time series
        that contain (at least) one FP observation.
    - estimates: contains the calculated estimates of mu, sigma and gamma, as
        parameters of the generalized extreme value distribution.

    :param methods: List containing names of compared methods.
    :return: Filepaths to log files.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath_pt = "../results/per_point_" + now + ".csv"
    filepath_ts = "../results/per_time_series_" + now + ".csv"
    filepath_est = "../results/estimates_" + now + ".csv"

    header_pt = ["Experiment ID"]
    for method in methods:
        header_pt.extend(
            [f"Empirical Rejection Rate ({method})",
             f"True Positives ({method})", f"False Negatives ({method})",
             f"False Positives ({method})", f"True Negatives ({method})"]
        )

    with open(filepath_pt, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header_pt)

    header_ts = ["Experiment ID"] + [f"Empirical Rejection Rate ({method})"
                                     for method in methods]

    with open(filepath_ts, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header_ts)

    header_estimates = [f"{val} {stat} ({mode})"
                        for mode in ['parallel', 'sequential']
                        for val in ['mu', 'sigma', 'gamma']
                        for stat in ['mean', 'std']]

    with open(filepath_est, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header_estimates)

    return filepath_pt, filepath_ts, filepath_est


def calculate_statistics(mode: str, results: dict, methods: list = None) -> list:
    """
    Auxiliary function to calculate statistics depending on mode.

    :param mode: Either of the following, given as string:
    - per_point: statistics are based on individual observations, e.g. the
        number of false positives (FPs) coincides with the average FPs over
        all time series.
    - per_time_series: statistics are based on aggregated time series, e.g. the
        number of false positives (FPs) coincides with the number of time series
        that contain (at least) one FP observation.
    - estimates: contains the calculated estimates of mu, sigma and gamma, as
        parameters of the generalized extreme value distribution.
    :param results: Results of an individual experiment.
    :param methods: List containing names of compared methods.
    :return: List containing the statistics.
    """
    result = []

    if mode == 'per_point':
        for method in methods:
            mean_rej_rate = np.mean(results[method][0])
            mean_cm = np.mean(results[method][1], axis=0)
            result.append(mean_rej_rate)
            result.extend(mean_cm.flatten().tolist())
    elif mode == 'per_time_series':
        result = [np.sum(results[method][0] > 0) / len(results[method][0])
                  for method in methods]
    elif mode == 'estimates':
        for method in ['parallel', 'sequential']:
            result.extend(list(results[f"Estimates ({method})"]))

    return result


if __name__ == '__main__':
    alternative_methods = ['Campulova2018', 'Holesovsky2018', 'Wette2024']
    modes = ['parallel', 'sequential', 'parallel (true quantile)',
             'sequential (true quantile)']
    methods = [f'Ours ({m})' for m in modes] + alternative_methods
    filepath_pt, filepath_ts, filepath_est = prepare_logfiles(methods)
    filepaths = [('per_point', filepath_pt),
                 ('per_time_series', filepath_ts),
                 ('estimates', filepath_est)]

    config = (get_config_general(False) + get_config_general(True)
              + get_config_outlier_height() + get_config_outlier_stable()
              + get_config_short(False) + get_config_short(True))

    for experiment_id, args, kwargs in config:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(now + f": Experiment: {experiment_id}")
        results = experiment(*args, **kwargs, methods=alternative_methods,
                             modes=modes)

        for mode, fp in filepaths:
            result = calculate_statistics(mode, results, methods)

            with open(fp, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([experiment_id] + result)
