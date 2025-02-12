#
# alternative_methods.py
#
# Project: Outlier Detection in Time Series
# Date: 2024-12-05
# Author: Florian Heinrichs
#
# Script containing implementations of the alternative methods:
# Campulova2018: Čampulová, M., Michálek, J., Mikuška, P., & Bokal, D. (2018).
#                Nonparametric algorithm for identification of outliers in
#                environmental data. Journal of Chemometrics, 32(5), e2997.
# Holevsovsky2018: Holešovský, J., Čampulová, M., & Michalek, J. (2018).
#                  Semiparametric outlier detection in nonstationary times
#                  series: Case study for atmospheric pollution in Brno, Czech
#                  Republic. Atmospheric Pollution Research, 9(1), 27-36.
# Wette2024: Wette, S., & Heinrichs, F. (2024). OML-AD: Online Machine Learning
#            for Anomaly Detection in Time Series Data. arXiv preprint
#            arXiv:2409.09742.
# Munir2018: Munir, M., Siddiqui, S. A., Dengel, A., & Ahmed, S. (2018).
#            DeepAnT: A deep learning approach for unsupervised anomaly
#            detection in time series. IEEE Access, 7, 1991-2005.
# Malhotra2015: Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015, April).
#               Long short term memory networks for anomaly detection in time
#               series. In ESANN (Vol. 2015, p. 89).

import numpy as np
from river import anomaly, linear_model, optim, preprocessing, time_series
import rpy2.robjects as ro
from scipy.stats import norm
import tensorflow as tf


ro.r('library(envoutliers)')


def campulova_2018(X: np.ndarray, alpha: float, n: int) -> np.ndarray:
    """
    Wrapper for the outlier detection method by Campulova et al. (2018)
    implemented in the corresponding R package 'envoutliers'
        https://cran.r-project.org/web/packages/envoutliers/envoutliers.pdf

    :param X: NumPy array of size (n_time_series, n_samples_per_ts).
    :param alpha: Level of the test.
    :param n: Number of observations considered as one block.
    :return: Test decision for each point as NumPy array of the same size as X.
    """
    if len(X.shape) == 1:
        X = X[np.newaxis, :]

    reject_null = np.zeros_like(X, dtype=bool)
    alpha_corrected = alpha / n
    alpha_float = ro.FloatVector([alpha_corrected])

    for i, X_i in enumerate(X):
        for j in range(n):
            data = ro.FloatVector(X_i[j:].tolist())

            ro.r(f"""
            result <- tryCatch(
                envoutliers::KRDetect.outliers.changepoint(
                    {data.r_repr()}, 'alpha.default'={alpha_float.r_repr()[0]}, 
                    'method'='chebyshev.inequality'
                ),
                error = function(e) NULL,  # Suppress any error
                warning = function(w) NULL  # Suppress any warning
            )
            """)

            result = ro.r['result']
            if result is not ro.NULL:
                result = result.rx2('outlier')
                break

        if result is not ro.NULL:
            reject_null[i, -len(result):] = np.array(result)

    return reject_null


def holesovsky_2018(X: np.ndarray, alpha: float, n: int) -> np.ndarray:
    """
    Wrapper for the outlier detection method by Holesovsky et al. (2018)
    implemented in the corresponding R package 'envoutliers'
        https://cran.r-project.org/web/packages/envoutliers/envoutliers.pdf

    :param X: NumPy array of size (n_time_series, n_samples_per_ts).
    :param alpha: Level of the test.
    :param n: Number of observations considered as one block.
    :return: Test decision for each point as NumPy array of the same size as X.
    """
    if len(X.shape) == 1:
        X = X[np.newaxis, :]

    reject_null = np.zeros_like(X, dtype=bool)
    return_period = n / alpha
    rp_float = ro.FloatVector([return_period])

    for i, X_i in enumerate(X):
        for j in range(n):
            data = ro.FloatVector(X_i[j:].tolist())

            ro.r(f"""
            result <- tryCatch(
                envoutliers::KRDetect.outliers.EV(
                    {data.r_repr()}, 'return.period'={rp_float.r_repr()[0]}
                ),
                error = function(e) NULL,  # Suppress any error
                warning = function(w) NULL  # Suppress any warning
            )
            """)

            result = ro.r['result']
            if result is not ro.NULL:
                result = result.rx2('outlier')
                break

        if result is not ro.NULL:
            reject_null[i, -len(result):] = np.array(result)

    return reject_null


def ml_based(
        X_stable: np.ndarray, X: np.ndarray, alpha: float, n: int,
        od_method: str, window_size: int = None, method: str = 'chebyshev'
) -> np.ndarray:
    """
    Wrapper for the ML-based outlier detection methods.

    :param X_stable: NumPy array containing the stable data, used for training,
        of size (n_time_series, n).
    :param X: NumPy array of size (n_time_series, n_samples_per_ts).
    :param alpha: Level of the test.
    :param n: Number of observations considered as one block. Should generally
        coincide with len(X_stable).
    :param od_method: Indicating outlier detection method to be used either of
        'Wette2024', 'Malhotra2015', 'Munir2018'.
    :param window_size: Number of observations used for predictions. If not
        specified, defaults to sqrt(n).
    :param method: Method to calculate quantile, either 'chebychev' or 'normal'.
    :return: Test decision for each point as NumPy array of the same size as X.
    """
    if len(X.shape) == 1:
        X = X[np.newaxis, :]

    if len(X_stable.shape) == 1:
        X_stable = X_stable[np.newaxis, :]

    if od_method == 'Wette2024':
        def single_test(x_stable, x):
            return wette_2024_single_ts(x_stable, x, alpha, n, method=method)

    elif od_method == 'Malhotra2015':
        def single_test(x_stable, x):
            return nn_based(x_stable, x, alpha, n, window_size=window_size,
                            method=method, model_type='LSTM')

    elif od_method == 'Munir2018':
        def single_test(x_stable, x):
            return nn_based(x_stable, x, alpha, n, window_size=window_size,
                            method=method, model_type='CNN')

    else:
        raise ValueError(f"{od_method} is not a valid outlier detection method")

    reject_null = np.stack([
        single_test(X_stable[i], X_i)
        for i, X_i in enumerate(X)
    ])

    return reject_null


def wette_2024_single_ts(
        X_stable: np.ndarray, X: np.ndarray, alpha: float, n: int,
        method: str = 'chebyshev'
) -> np.ndarray:
    """
    Implementation of the outlier detection method by Wette & Heinrichs (2024).
    Based on the corresponding GitHub repository:
        https://github.com/sebiwtt/OML-AD/tree/main

    :param X_stable: NumPy array containing the stable data, used for training,
        of size (n,).
    :param X: NumPy array of size (n_samples,).
    :param alpha: Level of the test.
    :param n: Number of observations considered as one block. Should generally
        coincide with len(X_stable).
    :param method: Method to calculate quantile, either 'chebychev' or 'normal'.
    :return: Test decision for each point as NumPy array of the same size as X.
    """
    X = np.concatenate([X_stable, X])
    warmup_period = len(X_stable)

    if method == 'chebyshev':
        n_std = np.sqrt(n / alpha)
    elif method == 'normal':
        n_std = norm.ppf(1 - alpha / n)
    else:
        raise ValueError(f"{method=} is not supported.")

    predictive_model = time_series.SNARIMAX(
        p=2, d=1, q=2, regressor=(
                preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(0.001), l2=0.01, intercept_lr=1e-10
        )
        ),
    )

    PAD = anomaly.PredictiveAnomalyDetection(
        predictive_model, horizon=1, n_std=n_std, warmup_period=warmup_period
    )

    scores = []

    for y in X:
        score = PAD.score_one(None, y)
        scores.append(score)
        PAD.learn_one(None, y)

    anomaly_scores = np.array(scores)[warmup_period:]
    reject_null = anomaly_scores >= 1

    return reject_null


def get_model(window_size: int, model_type: str = 'CNN') -> tf.keras.models.Model:
    """
    Method to create neural network to predict next time step for 'normal' data.

    :param window_size: Number of observations used for predictions.
    :param model_type: Either 'CNN' or 'LSTM'.
        - if 'CNN': Use model proposed by Munir et al. (2018)
        - if 'LSTM': Use model proposed by Malhotra et al. (2015)
    :return:
    """
    if model_type == 'CNN':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                   input_shape=(window_size, 1)),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mae')
    elif model_type == 'LSTM':
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(30, activation='sigmoid',
                                 input_shape=(window_size, 1),
                                 return_sequences=True),
            tf.keras.layers.LSTM(20, activation='sigmoid',
                                 return_sequences=False),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
    else:
        raise ValueError(f"{model_type=} is not supported.")

    return model


def nn_based(X_stable: np.ndarray, X: np.ndarray, alpha: float, n: int,
             window_size: int = None, method: str = 'chebyshev',
             model_type: str = 'CNN') -> np.ndarray:
    """
    Implementation of the neural network-based outlier detection methods for a
    single time series.

    :param X_stable: NumPy array containing the stable data, used for training,
        of size (n,).
    :param X: NumPy array of size (n_samples_per_ts,).
    :param alpha: Level of the test.
    :param n: Number of observations considered as one block. Should generally
        coincide with len(X_stable).
    :param window_size: Number of observations used for predictions. If not
        specified, defaults to sqrt(n).
    :param method: Method to calculate quantile, either 'chebychev' or 'normal'.
    :param model_type: Either 'CNN' or 'LSTM'.
        - if 'CNN': Use model proposed by Munir et al. (2018)
        - if 'LSTM': Use model proposed by Malhotra et al. (2015)
    :return: Test decision for each point as NumPy array of the same size as X.
    """
    if method == 'chebyshev':
        n_std = np.sqrt(n / alpha)
    elif method == 'normal':
        n_std = norm.ppf(1 - alpha / n)
    else:
        raise ValueError(f"{method=} is not supported.")

    if window_size is None:
        window_size = np.ceil(np.sqrt(n)).astype('int')

    window_size = max(window_size, 10)
    model = get_model(window_size, model_type=model_type)

    # Model training
    X_train = np.lib.stride_tricks.sliding_window_view(
        X_stable, window_shape=(window_size,))[:-1]
    y_train = X_stable[-len(X_train):]
    X_train = np.expand_dims(X_train, axis=-1)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0,
              callbacks=[early_stopping])

    y_stable = model.predict(X_train, verbose=0)
    residuals_stable = y_train - y_stable.squeeze()
    threshold = n_std * np.std(residuals_stable)
    mu = np.mean(residuals_stable)

    # Inference
    y_prep = X
    X = np.concatenate([X_stable[-window_size:], X])
    X_prep = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=(window_size,))[:-1]
    X_prep = np.expand_dims(X_prep, axis=-1)

    y_pred = model.predict(X_prep, verbose=0)
    residuals = y_prep - y_pred.squeeze()
    anomalies = np.abs(residuals - mu) > threshold

    del residuals_stable, residuals
    del X_train, y_train, model, y_stable, y_prep, X_prep

    return anomalies


if __name__ == '__main__':
    x = np.random.randn(2, 100)
    x[:, :50] += 1
    x[0, 25] += 3

    rej_null = campulova_2018(x, alpha=0.05, n=20)
    print(rej_null.astype(int))

    rej_null = holesovsky_2018(x, alpha=0.05, n=20)
    print(rej_null.astype(int))

    for od_method in ['Wette2024', 'Malhotra2015', 'Munir2018']:
        rej_null = ml_based(x[:, :20], x[:, 20:], alpha=0.05, n=20,
                            od_method=od_method, method='normal')
        print(od_method)
        print(rej_null.astype(int))