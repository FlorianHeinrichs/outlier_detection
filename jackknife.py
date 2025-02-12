#
# jackknife.py
#
# Project: Outlier Detection in Time Series
# Date: 2024-09-18
# Author: Florian Heinrichs
#
# Script to estimate the mean through local linear estimation.
# More details in:
# - "Bücher, A., Dette, H., & Heinrichs, F. (2021). Are deviations in a
# gradually varying mean relevant? A testing approach based on sup-norm
# estimators. The Annals of Statistics, 49(6), 3583-3617."

import numpy as np


def nadaraya_watson_estimation(X: np.ndarray, bw: int | np.ndarray,
                               filter_array: np.ndarray = None) -> np.ndarray:
    """
    Function to calculate the Nadaraya-Watson estimator. If multiple time series
    are provided, the calculations are done in parallel.

    :param X: Time series given as numpy array (possibly several).
    :param bw: Bandwidth of the estimator (possibly several).
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the NW-estimator(s) given as numpy array.
    """

    if filter_array is None:
        filter_array = np.ones(X.shape[-1], dtype=bool)
    X_filtered = X.copy()
    X_filtered[:, ~filter_array] = 0

    expand_dims = len(X_filtered.shape) == 1

    if expand_dims:
        X_filtered = X_filtered[np.newaxis, :]

    if isinstance(bw, (int, np.int32)):
        bw = np.array([bw] * len(X_filtered))

    _, kernel = get_kernel(bw)

    support = np.ones_like(X_filtered)
    support[:, ~filter_array] = 0

    padding = ((0, 0), (np.max(bw), 0))
    support = np.pad(support, padding, mode='edge')
    X_filtered = np.pad(X_filtered, padding, mode='edge')

    R = np.array([np.convolve(X_filt, kern[::-1], mode='valid')
                  for X_filt, kern in zip(X_filtered, kernel)])
    S = np.array([np.convolve(supp, kern[::-1], mode='valid')
                  for supp, kern in zip(support, kernel)]) + 1e-10
    estimator = (R / S)

    if expand_dims:
        estimator = estimator[0, :]

    return estimator


def bandwidth_cv(X: np.ndarray, num_folds: int, min_bw: int, max_bw: int,
                 step_size: int = 1) -> np.ndarray:
    """
    Function to tune the bandwidth of the local linear estimator. If multiple
    time series are provided, the calculations are done in parallel.

    :param X: Time series given as numpy array (possibly several).
    :param num_folds: Number of folds used for cross validation.
    :param min_bw: Minimal bandwidth of the estimator.
    :param max_bw: Maximal bandwidth of the estimator.
    :param step_size: Step size of bandwidth. Defaults to 1.
    :return: Returns the optimal bandwidth(s) given as numpy array.
    """
    indices = np.arange(X.shape[-1])
    indices_copy = indices.copy()
    np.random.shuffle(indices_copy)
    folds = np.split(indices_copy, num_folds)
    n_samples = X.shape[:-1] if len(X.shape) > 1 else 1

    best_bw = - np.ones(n_samples, dtype=int), - np.ones(n_samples)

    for bw in range(min_bw, max_bw + 1, step_size):
        mse = np.zeros(n_samples)
        for fold in folds:
            filter_array = ~np.isin(indices, fold)

            nw_estimator = nadaraya_watson_estimation(X, bw, filter_array)

            mse += np.nanmean(
                (X[..., ~filter_array] - nw_estimator[..., ~filter_array]) ** 2,
                axis=-1
            )

        better_bw = np.where((mse < best_bw[1]) | (best_bw[1] == -1))
        best_bw[0][better_bw], best_bw[1][better_bw] = bw, mse[better_bw]

    return best_bw[0]


def full_lle(X: np.ndarray, bw: int | np.ndarray) -> np.ndarray:
    """
    Function to calculate the local linear estimator. If multiple time series
    are provided, the calculations are done in parallel.

    :param X: Time series given as numpy array (possibly several).
    :param bw: Bandwidth of the estimator (possibly several).
    :return: Returns the local linear estimator(s) for the full time series.
    """
    expand_dims = len(X.shape) == 1

    if expand_dims:
        X = X[np.newaxis, :]

    if isinstance(bw, (int, np.int32)):
        bw = np.array(X.shape[0] * [bw])

    support, kernel = get_kernel(bw)
    S0 = np.sum(kernel, axis=-1, keepdims=True)
    S1 = np.sum(support * kernel, axis=-1, keepdims=True)
    S2 = np.sum(support ** 2 * kernel, axis=-1, keepdims=True)

    padding = ((0, 0), (np.max(bw), 0))
    X_pad = np.pad(X, padding, mode='edge')

    R0 = np.array([np.convolve(y, kern[::-1], mode='valid')
                   for y, kern in zip(X_pad, kernel)])
    R1 = np.array([np.convolve(y, supp_kern[::-1], mode='valid')
                   for y, supp_kern in zip(X_pad, support * kernel)])

    local_linear_estimator = (S2 * R0 - S1 * R1) / (S0 * S2 - S1 ** 2 + 1e-10)

    if expand_dims:
        local_linear_estimator = local_linear_estimator[0, :]

    return local_linear_estimator


def single_lle(X: np.ndarray, bw: int | np.ndarray,
               mask: np.ndarray) -> np.ndarray:
    """
    Function to calculate the local linear estimator. If multiple time series
    are provided, the calculations are done in parallel.

    :param X: Time series given as numpy array (possibly several).
    :param bw: Bandwidth of the estimator (possibly several).
    :param mask: Observations to be masked during estimation.
    :return: Returns the local linear estimator(s) for a single time step.
    """
    if isinstance(bw, (int, np.int32)):
        bw = np.array(X.shape[0] * [bw])

    expand_dims = len(X.shape) == 1

    if expand_dims:
        X = X[np.newaxis, :]

    if X.shape[-1] != np.max(bw) + 1:
        raise ValueError("Length of time series must match maximal bandwidth:"
                         f"{X.shape[-1]=}, {np.max(bw)=}.")

    support, kernel = get_kernel(bw)

    X_masked = X.copy()
    X_masked[mask] = 0
    support[mask] = 0
    kernel[mask] = 0

    S0 = np.sum(kernel, axis=-1, keepdims=True)
    S1 = np.sum(support * kernel, axis=-1, keepdims=True)
    S2 = np.sum(support ** 2 * kernel, axis=-1, keepdims=True)

    R0 = np.sum(X_masked * kernel, axis=-1, keepdims=True)
    R1 = np.sum(X_masked * support * kernel, axis=-1, keepdims=True)

    local_linear_estimator = (S2 * R0 - S1 * R1) / (S0 * S2 - S1 ** 2 + 1e-10)

    if expand_dims:
        local_linear_estimator = local_linear_estimator[0]

    return local_linear_estimator


def jackknife_estimation(X: np.ndarray, bw: int | np.ndarray,
                         mask: np.ndarray = None) -> np.ndarray:
    """
    Function to calculate the Jackknife version of the local linear estimator.
    If multiple time series are provided, the calculations are done in parallel.

    :param X: Time series given as numpy array (possibly several).
    :param bw: Bandwidth of the estimator (possibly several).
    :param mask: Observations to be masked during estimation. If provided, only
        a single prediction is made and its shape must match the shape of X.
    :return: Returns the local linear estimator(s) given as numpy array.
    """
    if isinstance(bw, (int, np.int32)):
        bw = np.array(X.shape[0] * [bw], dtype=int)

    if mask is None:
        jackknife_estimate = (
                2 * full_lle(X, (bw // np.sqrt(2)).astype(int))
                - full_lle(X, bw))
    elif mask.shape == X.shape and mask.shape[-1] == np.max(bw) + 1:
        bw_2 = (bw // np.sqrt(2)).astype(int)
        start_idx = - np.max(bw_2) - 1
        jackknife_estimate = (
                2 * single_lle(X[:, start_idx:], bw_2, mask[:, start_idx:])
                - single_lle(X, bw, mask))
    elif mask.shape[-1] != np.max(bw) + 1:
        raise ValueError(f"Shape of mask does not match maximum bandwidth: + 1"
                         f"{mask.shape[-1]=}, {np.max(bw) + 1=}")
    else:
        raise ValueError(f"If mask is provided, only a single prediction is "
                         f"made. Shapes must match: {mask.shape=}, {X.shape=}")

    return jackknife_estimate


def get_kernel(bw: int | np.ndarray,
               mode: str = 'quartic') -> (np.ndarray, np.ndarray):
    """
    Auxiliary function to define the kernel.

    :param bw: Bandwidth of the estimator (possibly several).
    :param mode: Mode of the kernel given as string. Currently, only the quartic
        kernel is supported.
    :return: Returns the kernel and its support as numpy arrays.
    :raises: ValueError if unsupported mode is chosen.
    """
    bw_is_int = isinstance(bw, (int, np.int32))
    bw = np.array([bw]) if bw_is_int else bw

    support = [np.arange(-h, 1) / h for h in bw]

    if mode == 'quartic':
        kernel = [15 / 16 * (1 - supp ** 2) ** 2 for supp in support]
    elif mode == 'triweight':
        kernel = [35 / 32 * (1 - supp ** 2) ** 3 for supp in support]
    elif mode == 'tricube':
        kernel = [70 / 81 * (1 - np.abs(supp) ** 3) ** 3 for supp in support]
    else:
        raise ValueError(f"{mode=} unknown.")

    support = stack_var(support)
    kernel = stack_var(kernel)

    if bw_is_int:
        support, kernel = support[0], kernel[0]

    return support, kernel


def stack_var(x: list) -> np.ndarray:
    """
    Stack 1-dimensional NumPy arrays of varying length.

    :param x: List with NumPy arrays of different lengths.
    :return: Stacked NumPy arrays.
    """
    max_len = max(len(y) for y in x)
    x_padded = [np.pad(y, (max_len - len(y), 0), mode='constant') for y in x]

    return np.stack(x_padded, axis=0)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 500
    signal = np.sin(np.linspace(0, 3 * 2 * np.pi, n))[None, :]
    noise = np.random.randn(1, n) / 10
    X = signal + noise

    bw = bandwidth_cv(X, 5, 2, 200)

    mu_nwe = nadaraya_watson_estimation(X, bw)
    mu_lle = full_lle(X, bw)
    mu_jack = jackknife_estimation(X, bw)

    max_bw = np.max(bw)
    mask = np.zeros((X.shape[0], max_bw + 1), dtype=bool)
    X_pad = np.pad(X, ((0, 0), (max_bw, 0)), mode='edge')
    predictions = []

    for timestep in range(n):
        pred = jackknife_estimation(X_pad[:, timestep: timestep + (max_bw + 1)],
                                    bw, mask=mask)
        predictions.append(pred)

    mu_jack_sequential = np.concatenate(predictions, axis=-1)

    all_estimators = [(X, 'raw'), (mu_nwe, 'nwe'),  (mu_lle, 'lle'),
                      (mu_jack, 'jackknife'),
                      (mu_jack_sequential, 'sequential_jackknife')
                      ]

    plt.plot(np.arange(n), signal[0], label='mu')
    for data, title in all_estimators:
        plt.plot(data[0], label=title)

    plt.legend()
    plt.show()

