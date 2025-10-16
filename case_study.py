#
# case_study.py
#
# Project: Outlier Detection in Time Series
# Date: 2025-01-18
# Author: Florian Heinrichs
#
# Main script containing real data experiments.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from tueplots import bundles, figsizes

from experiments import parallel_test, sequential_test, alternative_tests
from jackknife import bandwidth_cv, full_lle, single_lle
from real_data_experiment import load_data, impute_nans

plt.rcParams.update(bundles.icml2022(family="Times New Roman", usetex=False))
plt.rcParams.update(figsizes.icml2022_full())


def experiment(data: np.ndarray, m_stable: int,
               quantile_kwargs: dict = None,
               cv_kwargs: dict = None,
               methods: list = None,
               return_params: bool = True) -> tuple:
    """
    Main function for running the experiments.

    :param data: Time series to be analysed for outliers.
    :param m_stable: Length of stable period without outliers.
    :param quantile_kwargs: Dictionary containing settings for quantile
        estimation.
    :param cv_kwargs: Dictionary containing settings for cross-validation to
        optimize the jackknife estimator's bandwidth.
    :param methods: List of alternative methods that should be used.
    :param return_params: Parameter specifying whether to return tuned bandwidth.
    :return: Empirical rejection rate as float and confusion matrix as NumPy
        array.
    """
    if methods is None:
        methods = ['Campulova2018','Holesovsky2018', 'Wette2024']

    if cv_kwargs is None:
        cv_kwargs = {'num_folds': 5, 'min_bw': 30, 'max_bw': 100}

    bw = cv_kwargs.get('bandwidth')

    if bw is None:
        bw = bandwidth_cv(data[..., :m_stable], **cv_kwargs)

    m_remaining = data.shape[-1] - m_stable
    mean_estimator = full_lle(data, bw)

    if quantile_kwargs is None:
        alpha = 0.05
        quantile_kwargs = {'alpha': alpha * np.ones(m_remaining),
                           'return_estimates': True}

    results = {}

    for mode in ['full', 'partial']:
        if mode == 'full':
            reject_null, mu, sigma, gamma = parallel_test(
                data, bw, m_stable, quantile_kwargs, estimation=full_lle
            )
        else:
            reject_null, mu, sigma, gamma = sequential_test(
                data, bw, m_stable, m_remaining, quantile_kwargs,
                estimation=single_lle)

        n_outliers = np.sum(reject_null, axis=-1)
        results[f"Ours ({mode})"] = (n_outliers, reject_null)

    results_alt = alternative_tests(data, m_stable, m_remaining, alpha,
                                    return_rejections=True, methods=methods)

    for method, result in results_alt.items():
        rej_null = result[-1]
        n_outliers = np.sum(rej_null, axis=-1)
        results_alt[method] = (n_outliers, rej_null)

    results.update(results_alt)

    if return_params:
        return results, mean_estimator, (bw, mu, sigma, gamma)
    else:
        return results, mean_estimator


def simple_case_study():
    data = load_data('Melbourne Regional Office')

    data, start_idx = impute_nans(data, initial_period=365, min_nans=1)
    data = data[start_idx:]
    date = data['Date'].to_numpy()

    temperature = data['Temperature'].to_numpy()
    results, mean_estimate = experiment(temperature[None, :], 365)

    segments = [('1867-07-01', '1869-06-30'), ('1967-07-01', '1968-06-30')]

    fig, axes = plt.subplots(1, 2)
    colors = ['tab:green', 'tab:red', 'tab:purple']
    markers = ['x', '+', "2"]

    for ax, (start, end) in zip(axes, segments):
        start, end = np.datetime64(start), np.datetime64(end)
        idx = np.where((date >= start) & (date <= end))[0]
        ax.plot(date[idx], temperature[idx], zorder=1)
        ax.plot(date[idx], mean_estimate[0, idx], zorder=2)

        for i, (method, res) in enumerate(results.items()):
            detected = np.where(res[1][0][idx - 365])[0] + idx[0]
            emp_rej_rate = res[0]
            print(f"{method}: {emp_rej_rate}")
            ax.scatter(date[detected], temperature[detected] + (2 - i),
                        label=method, c=colors[i], zorder=i+2, marker=markers[i])

    axes[1].legend()
    plt.show()


def extended_case_study(city: str,
                        print_outliers: int = -1,
                        initial_period: int = 365,
                        display_acf: bool = False,
                        return_outliers: bool = False) -> dict | tuple:
    data = load_data(city)
    data, start_idx = impute_nans(data, initial_period=initial_period, min_nans=1)
    data = data[start_idx:]
    temperature = data['Temperature'].to_numpy()

    if display_acf:
        plot_acf(temperature[:initial_period], lags=50)
        plt.show()

    results = {}
    outliers = {}

    for bw_selection in ['automatic', 'fixed']:
        if bw_selection == 'automatic':
            methods_tmp = ['Campulova2018', 'Holesovsky2018', 'Wette2024']
            cv_kwargs = {'num_folds': initial_period // 30, 'min_bw': 2,
                         'max_bw': 100, 'shuffle': False}
            postfix = ""
        else:
            methods_tmp = []
            cv_kwargs = {'bandwidth': np.array([7])}
            postfix = " (fixed_bw)"

        result, mean_estimate, params = experiment(
            temperature[None, :], initial_period, cv_kwargs=cv_kwargs,
            methods=methods_tmp
        )

        for method, res in result.items():
            n_outliers = res[0][0]
            results[method + postfix] = n_outliers
            outliers[method + postfix] = res[1]

        param_names = ['bandwidth', 'mu', 'sigma', 'gamma']
        for p_name, param in zip(param_names, params):
            results[p_name + postfix] = param.flat[0]

        for mode in ['full', 'partial']:
            if result[f'Ours ({mode})'][0][0] <= print_outliers and bw_selection == 'automatic':
                idx = np.where(result[f'Ours ({mode})'][1] > 0)[1] + initial_period
                outlier_dates = data.iloc[idx]['Date'].tolist()
                print(f"{bw_selection} bandwidth - {mode=}: {outlier_dates=}")

    results['n_total'] = len(data)

    return (results, outliers) if return_outliers else results

def extended_case_study_plots(n_init: int = 730,
                              cities: list = None,
                              methods: list = None,
                              labels: list = None):
    if cities is None:
        cities = ['Gayndah Post Office', 'Gunnedah Pool']

    if methods is None:
        methods = ['Ours (full)', 'Ours (full) (fixed_bw)', 'Campulova2018']

    if labels is None:
        labels = ['Full ($h^*$)', "Full ($h_{fix}$)", "Ca2018"]

    colors = ['tab:orange', 'tab:red', 'tab:green']
    markers = ['x', "2", '+']

    fig, axes = plt.subplots(1, len(cities))

    for ax, city in zip(axes, cities):
        data = load_data(city)
        data, start_idx = impute_nans(data, initial_period=n_init, min_nans=1)
        data = data[start_idx:]
        temperature = data['Temperature'].to_numpy()
        date = data['Date'].to_numpy()
        outliers = extended_case_study(city, initial_period=n_init, return_outliers=True)[1]

        start, end = np.datetime64('2003-09-01'), np.datetime64('2005-08-31')
        idx = np.where((date >= start) & (date <= end))[0]

        ax.plot(date[idx], temperature[idx], c='tab:blue', zorder=1)
        ax.set_title(city)

        for m, (color, marker, method, label) in enumerate(zip(colors, markers, methods, labels)):
            outlier_positions = np.where(outliers[method][0])[0]  + n_init
            idx_tmp = [i for i in idx if i in outlier_positions]
            temp = temperature[idx_tmp] + m / 2
            ax.scatter(date[idx_tmp], temp, c=color,
                       label=label, zorder=m+2, marker=marker)

    axes[0].legend()
    plt.show()


if __name__ == '__main__':
    cities = ['Boulia Airport', 'Gayndah Post Office', 'Gunnedah Pool',
              'Hobart TAS', 'Melbourne Regional Office',
              'Cape Otway Lighthouse', 'Robe', 'Sydney']
    all_results = {city: extended_case_study(city) for city in cities}

    df = pd.DataFrame.from_dict(all_results, orient='index')
    print(df)

    # extended_case_study_plots(methods = ['Ours (full)', 'Campulova2018'],
    #                           labels=['Full ($h^*$)', "Ca2018"])
