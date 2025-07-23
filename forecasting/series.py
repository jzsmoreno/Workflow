import numpy as np
import pandas as pd
from likelihood.tools import *
from tensorflow.keras.models import load_model


def create_index(data_series, start="2018"):
    n = data_series.shape[1]
    stop = np.datetime64("2018") + np.timedelta64(n, "M")
    date = np.arange(start + "-01", stop, dtype="datetime64[M]")
    return date


def convert_to_df(data_series):
    return pd.DataFrame(data_series[:, :].T, index=create_index(data_series))


def create_train_test_set(data_series, p_train=0.7, steps=4):
    """
    Divide un conjunto de series temporales (2D) en entrenamiento y prueba, separando por filas (series).

    Parameters
    ----------
    - data_series: ndarray o array-like de forma (n_series, n_steps)
    - p_train: float, proporción de series usadas para entrenamiento
    - steps: int, número de pasos a predecir (al final de cada serie)

    Returns
    ----------
    - train: [x_train, y_train]
    - test: [x_test, y_test]
    """
    n_series, n_steps = data_series.shape

    n_train = int(n_series * p_train)

    # Entrenamiento
    x_train = data_series[:n_train, :-steps]
    y_train = data_series[:n_train, -steps:]

    # Prueba
    x_test = data_series[n_train:, :-steps]
    y_test = data_series[n_train:, -steps:]

    return [x_train, y_train], [x_test, y_test]


def forecasting(model, train, m, scaler, steps=4.0):
    n = int(train.shape[1])
    n_periods = int((m - n) / steps)
    data_pred = np.copy(train)

    for step_ahead in range(n_periods + 1):
        y_pred_one = model.predict(data_pred[:, step_ahead:], verbose=0)
        data_pred = np.concatenate([data_pred, y_pred_one[:, :, np.newaxis]], axis=1)
    series_pred = scaler.scale(np.copy(data_pred[:, :, 0]))
    return series_pred
