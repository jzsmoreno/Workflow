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
    n_steps = data_series.shape[1]

    train_ = int(data_series.shape[0])
    train_init = int(data_series.shape[1] * p_train)
    x_train = data_series[:train_, : train_init - steps]
    y_train = data_series[:train_, train_init - steps : train_init]

    train = [x_train, y_train]

    x_test = data_series[train_:, : train_init - steps]
    y_test = data_series[train_:, train_init - steps : train_init]

    test = [x_test, y_test]

    return train, test


def forecasting(model, train, m, values, steps=4.0):
    n = int(train[0].shape[1])
    n_periods = int((m - n) / steps)
    data_pred = np.copy(train[0])[:, 0:n]

    for step_ahead in range(n_periods + 1):
        y_pred_one = model.predict(data_pred[:, step_ahead : data_pred.shape[1]])
        data_pred = np.concatenate([data_pred, y_pred_one], axis=1)
    series_pred = scale(np.copy(data_pred), values)
    return series_pred
