#
# experiments.py
#
# Project: Outlier Detection in Time Series
# Date: 2025-01-18
# Author: Florian Heinrichs
#
# Main script containing real data experiments.

import numpy as np
import pandas as pd

from jackknife import bandwidth_cv
from experiments import parallel_test, sequential_test, alternative_tests

def load_data(city: str) -> pd.DataFrame:
    codes = {'Boulia Airport': '038003',
             'Gayndah Post Office': '039039',
             'Gunnedah Pool': '055023',
             'Hobart TAS': '094029',
             'Melbourne Regional Office': '086071',
             'Cape Otway Lighthouse': '090015',
             'Robe': '026026',
             'Sydney': '066062'}

    code = codes[city]
    filepath = f"data/IDCJAC0010_{code}_1800_Data.csv"
    df = pd.read_csv(filepath)
    data = df.iloc[:, 2:6]
    data.rename(columns={'Maximum temperature (Degree C)': 'Temperature'},
                inplace=True)

    return data

def experiment(data: np.ndarray, m_stable: int,
               quantile_kwargs: dict = None,
               cv_kwargs: dict = None) -> dict:
    """
    Main function for running the experiments.

    :param data: Time series to be analysed for outliers.
    :param m_stable: Length of stable period without outliers.
    :param quantile_kwargs: Dictionary containing settings for quantile
        estimation.
    :param cv_kwargs: Dictionary containing settings for cross-validation to
        optimize the jackknife estimator's bandwidth.
    :return: Empirical rejection rate as float and confusion matrix as NumPy
        array.
    """
    if cv_kwargs is None:
        cv_kwargs = {'num_folds': 5, 'min_bw': 30, 'max_bw': 100}

    # Generate Data
    bw = bandwidth_cv(data[..., :m_stable], **cv_kwargs)
    m_remaining = data.shape[-1] - m_stable

    if quantile_kwargs is None:
        alpha = 0.01
        quantile_kwargs = {'alpha': alpha * np.ones(m_remaining)}

    results = {}

    for mode in ['parallel', 'sequential']:

        if mode == 'parallel':
            reject_null = parallel_test(data, bw, m_stable, quantile_kwargs)
        else:
            reject_null = sequential_test(data, bw, m_stable, m_remaining,
                                          quantile_kwargs)

        empirical_rejection_rate = np.sum(reject_null, axis=-1) / m_remaining

        results[f"Ours ({mode})"] = (empirical_rejection_rate, reject_null)

    alpha = float(quantile_kwargs['alpha'][0])
    alternatives = alternative_tests(data, m_stable, m_remaining, alpha,
                                     return_rejections=True)
    results.update(alternatives)

    return results


def find_nan_segments(data):
    mask = np.isnan(data).astype(int)
    segments = []

    if np.any(mask):
        mask_pad = np.concatenate(([False], mask, [False]))
        starts = np.where(np.diff(mask_pad) == 1)[0]
        ends = np.where(np.diff(mask_pad) == -1)[0] - 1

        segments = [(start, end, end - start + 1)
                    for start, end in zip(starts, ends)]

    return segments


def impute_nans(data, initial_period: int = 365, min_nans: int = 3) -> tuple:
    avg_temp = data.groupby(['Day', 'Month'])['Temperature'].mean().reset_index()
    avg_temp.rename(columns={'Temperature': 'Tmp'}, inplace=True)
    data = data.merge(avg_temp, on=['Day', 'Month'], how='left')

    data['Missing'] = data['Temperature'].isna()
    data['segment'] = (data['Missing'] != data['Missing'].shift()).cumsum()

    def impute_consecutive_missing(df):
        if df['Missing'].sum() >= min_nans:
            df.loc[df['Missing'], 'Temperature'] = df.loc[df['Missing'], 'Tmp']
        return df

    for segment, df in data.groupby('segment'):
        if df['Missing'].sum() == 0 and len(df) >= initial_period:
            start_idx = df.index[0]
            break
    else:
        start_idx = -1

    data = data.groupby('segment').apply(impute_consecutive_missing)
    data.reset_index(drop=True, inplace=True)
    data.drop(columns=['segment'], inplace=True)

    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    data['Outlier'] = data['Temperature'].isna()

    def impute_outliers(row, df):
        if pd.isna(row['Temperature']):
            current_date = row['Date']
            start = current_date - pd.Timedelta(days=365)

            last_year = df[(df['Date'] < current_date) & (df['Date'] >= start)]
            x = last_year['Temperature'].max() - last_year['Temperature'].min()
            y = row['Tmp']
            z = last_year['Temperature'].mean()

            temp = y - x if y > z else y + x
        else:
            temp = row['Temperature']

        return temp

    data['Temperature'] = data.apply(
        lambda row: impute_outliers(row, data), axis=1)

    data.drop(columns=['Tmp', 'Year', 'Month', 'Day'], inplace=True)

    return data, start_idx


def get_confusion_matrices(cities: list, m_stable: int) -> dict:
    all_results = {}

    for city in cities:
        data = load_data(city)
        data, start_idx = impute_nans(data)

        if start_idx == -1:
            print(f"No segment without outliers of length {m_stable} found.")
        else:
            print(f"{city} - Start index: {start_idx}")

        temperature = data['Temperature'].to_numpy().transpose()[start_idx:]
        results = experiment(temperature[None, :], m_stable)

        outliers = np.where(data['Outlier'].to_numpy()[start_idx:])
        result_city = {}

        for method, res in results.items():
            detected = np.where(res[1][0])[0] + m_stable
            x_11 = len(np.intersect1d(outliers[0], detected))
            x_12 = len(detected) - x_11
            x_21 = len(outliers[0]) - x_11
            x_22 = len(temperature) - x_11 - x_12 - x_21
            cm = pd.DataFrame([[x_11, x_12], [x_21, x_22]],
                              columns=['Outlier', 'Normal'],
                              index=['Reject Null', 'Accept Null'])

            result_city[method] = res[0], cm

        all_results[city] = result_city

    return all_results


if __name__ == '__main__':
    cities = ['Boulia Airport', 'Gayndah Post Office', 'Gunnedah Pool',
              'Hobart TAS', 'Melbourne Regional Office',
              'Cape Otway Lighthouse', 'Robe', 'Sydney']
    m_stable = 365
    results = get_confusion_matrices(cities, m_stable)

    print(results)
