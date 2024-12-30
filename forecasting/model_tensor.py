import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from likelihood.tools import *


class SimpleNetRNN(tf.keras.models.Model):
    def __init__(self, n_neurons):
        super(SimpleNetRNN, self).__init__()
        self.n_neurons = n_neurons
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.SimpleRNN(
                    self.n_neurons, return_sequences=True, input_shape=[None, 1]
                ),
                tf.keras.layers.SimpleRNN(self.n_neurons),
                tf.keras.layers.Dense(4, activation="tanh"),
            ]
        )

    def call(self, x):
        x = self.network(x)
        return x


class SimpleNetGRU(tf.keras.models.Model):
    def __init__(self, filters, kernel_size, n_units, strides):
        super(SimpleNetGRU, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.n_units = n_units
        self.strides = strides
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding="valid",
                    input_shape=[None, 1],
                ),
                tf.keras.layers.GRU(self.n_units, return_sequences=True),
                tf.keras.layers.GRU(self.n_units),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )

    def call(self, x):
        x = self.network(x)
        return x


class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


def make_predictions(model, input_model, n_steps):
    X = input_model.copy()
    for step_ahead in range(n_steps):
        y_pred_one = model.predict(X[:, step_ahead:])
        y_pred_one = y_pred_one[:, :, np.newaxis]
        X = np.concatenate((X, y_pred_one), axis=1)

    return X[:, :, 0]


if __name__ == "__main__":
    num_series = 80
    series_size = 100
    n_steps = 5

    input_serie = generate_series(num_series, series_size, incline=False)
    y_new = input_serie[:, :-n_steps]
    print(y_new.shape)
    scaler = DataScaler(y_new)
    y_new = scaler.rescale()
    size_ = int(0.8 * y_new.shape[0])
    x_train = y_new[:size_, :-4]
    y_train = y_new[:size_, -4:]
    x_test = y_new[size_:, :-4]
    y_test = y_new[size_:, -4:]

    print(x_train.shape, y_train.shape)

    # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model = SimpleNetRNN(n_neurons=10)
    # model = SimpleNetGRU(filters = 10, kernel_size = 4, n_units = 5, strides = 2)
    # model(x_train[:, :, np.newaxis])
    model.compile(loss="mse", optimizer="sgd", metrics=["mae"])
    history = model.fit(
        x_train[:, :, np.newaxis], y_train, epochs=100, validation_split=0.2, callbacks=[PrintDot()]
    )

    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    hist.tail()

    # plot history
    plt.plot(history.history["loss"], label="Train Error")
    plt.plot(history.history["val_loss"], label="Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    X = make_predictions(model, x_train[:, :, np.newaxis], n_steps=n_steps)
    print(X.shape)
    X = scaler.scale(X)

    plt.plot(range(len(input_serie[0, :])), input_serie[0, :], "o-", label="real value")
    plt.plot(range(len(X[0, :]))[-n_steps * 4 :], X[0, :][-n_steps * 4 :], "-r", label="prediction")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # savel model
    # model.save("./models/model_tensor", save_format="tf")
    model.network.save("./models/model_tensor", save_format="tf")
