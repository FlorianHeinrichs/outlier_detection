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

import numpy as np

from alternative_methods import campulova_2018, holesovsky_2018, ml_based
from data_generation import generate_data
from experiments_config import get_config
from jackknife import bandwidth_cv, jackknife_estimation
from quantiles import get_quantile


def parallel_test(data: np.ndarray, bw: int | np.ndarray, m_stable: int,
                  quantile_kwargs: dict) -> np.ndarray:
    """
    Auxiliary function to conduct parallel tests.

    :param data: NumPy array containing the data, of size
        (n_time_series, n_samples_per_ts).
    :param bw: Bandwidth to use for kernel regression (can vary per time series).
    :param m_stable: Length of stable period without outliers.
    :param quantile_kwargs: Dictionary containing keyword arguments for
        quantiles.
    :return: NumPy array containing test decisions.
    """
    mean_estimator = jackknife_estimation(data, bw)
    residuals = data - mean_estimator

    residuals_stable = residuals[..., :m_stable]
    quantile_estimator = get_quantile(residuals_stable, **quantile_kwargs)

    residuals = residuals[..., m_stable:]
    reject_null = np.abs(residuals) > quantile_estimator

    return reject_null

def sequential_test(data: np.ndarray, bw: int | np.ndarray, m_stable: int,
                    m_remaining: int, quantile_kwargs: dict) -> np.ndarray:
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
    :return: NumPy array containing test decisions.
    """
    n_time_series = data.shape[0]

    mean_estimator = jackknife_estimation(data[..., :m_stable], bw)
    residuals_stable = data[..., :m_stable] - mean_estimator
    quantile_estimator = get_quantile(residuals_stable, **quantile_kwargs)

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

        mean_estimator = jackknife_estimation(current_data, bw, mask=mask)
        residuals = (current_data - mean_estimator)[:, -1]

        reject_null_t = np.abs(residuals) > quantile_estimator[:, time_step]
        reject_null[:, time_step] = reject_null_t

    return reject_null

def alternative_tests(data: np.ndarray, m_stable: int, m_remaining: int,
                      alpha: float, outliers_index: np.ndarray = None,
                      return_rejections: bool = False) -> dict:
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
    :return: Dictionary containing test results.
    """
    results = {}

    alternatives = [('Campulova2018', campulova_2018),
                    ('Holesovsky2018', holesovsky_2018),
                    ('Wette2024', None), ('Malhotra2015', None),
                    ('Munir2018', None)]

    for od_method, func in alternatives:
        if func is None:
            rej_null = ml_based(data[:, :m_stable], data[:, m_stable:],
                                alpha=alpha, n=m_stable, od_method=od_method,
                                method='chebyshev')
        else:
            rej_null = func(data, alpha=alpha, n=m_stable)[:, m_stable:]

        emp_rej_rate = np.sum(rej_null, axis=-1) / m_remaining

        if outliers_index is None:
            results[od_method] = emp_rej_rate
        else:
            cm = calculate_confusion_matrix(outliers_index - m_stable, rej_null)
            results[od_method] = emp_rej_rate, cm

        if return_rejections:
            results[od_method] = *results[od_method], rej_null

    return results

def experiment(data_kwargs: dict,
               m_stable: int,
               cv_kwargs: dict,
               quantile_kwargs: dict,
               outlier_kwargs: dict = None) -> dict:
    """
    Main function for running the experiments.

    :param data_kwargs: Dictionary containing settings for data generation.
    :param m_stable: Length of stable period without outliers.
    :param cv_kwargs: Dictionary containing settings for cross-validation to
        optimize the jackknife estimator's bandwidth.
    :param quantile_kwargs: Dictionary containing settings for quantile
        estimation.
    :param outlier_kwargs: Dictionary containing settings for outlier generation.
    :return: Empirical rejection rate as float and confusion matrix as NumPy
        array.
    """
    # Generate Data
    data = generate_data(**data_kwargs)
    bw = bandwidth_cv(data[..., :m_stable], **cv_kwargs)
    m_remaining = data.shape[-1] - m_stable
    n_time_series = data.shape[0]

    # Generate Outliers
    if outlier_kwargs:
        min_height = outlier_kwargs.get('min_height', 0)
        n_outliers = outlier_kwargs.get('n_outliers', 0)
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
    else:
        outliers_index = np.zeros(0, dtype=np.int32)

    results = {}

    for mode in ['parallel', 'sequential']:

        if mode == 'parallel':
            reject_null = parallel_test(data, bw, m_stable, quantile_kwargs)
        else:
            reject_null = sequential_test(data, bw, m_stable, m_remaining,
                                          quantile_kwargs)

        empirical_rejection_rate = np.sum(reject_null, axis=-1) / m_remaining

        confusion_matrix = calculate_confusion_matrix(
            outliers_index - m_stable, reject_null)

        results[f"Ours ({mode})"] = empirical_rejection_rate, confusion_matrix

    alpha = float(quantile_kwargs['alpha'][0])
    alternatives = alternative_tests(data, m_stable, m_remaining, alpha,
                                     outliers_index)
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

if __name__ == '__main__':
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    filepath = "../results/" + filename

    methods = ['Ours (parallel)', 'Ours (sequential)', 'Campulova2018',
               'Holesovsky2018', 'Wette2024',
               'Malhotra2015', 'Munir2018'
               ]

    header = ["Experiment ID", "Expected False Positives"]
    for method in methods:
        header.extend(
            [f"Empirical Rejection Rate ({method})",
             f"True Positives ({method})", f"False Negatives ({method})",
             f"False Positives ({method})", f"True Negatives ({method})"]
        )

    # with open(filepath, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)

    config = get_config(False) + get_config(True)
    for experiment_id, args, kwargs in config[3:]:
        results = experiment(*args, **kwargs)

        alpha = args[3].get('alpha', 0)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(now + f": Experiment: {experiment_id}")

        result = [experiment_id, np.sum(alpha)]

        for method in methods:
            mean_rej_rate = np.mean(results[method][0])
            mean_cm = np.mean(results[method][1], axis=0)
            result.append(mean_rej_rate)
            result.extend(mean_cm.flatten().tolist())

        print(result)

        # with open(filepath, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(result)