import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from likelihood.tools import *


class SimpleNetRNN(tf.keras.models.Model):
    def __init__(self, n_neurons, output_units):
        super(SimpleNetRNN, self).__init__()
        self.n_neurons = n_neurons
        self.output_units = output_units
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.SimpleRNN(
                    self.n_neurons, return_sequences=True, input_shape=[None, 1]
                ),
                tf.keras.layers.SimpleRNN(self.n_neurons),
                tf.keras.layers.Dense(self.output_units, activation="tanh"),
            ]
        )

    def call(self, x):
        x = self.network(x)
        return x


class SimpleNetGRU(tf.keras.Model):
    def __init__(
        self,
        filters=64,
        kernel_size=3,
        n_units=128,
        strides=1,
        output_units=1,
        dropout_rate=0.3,
        recurrent_dropout_rate=0.3,
        l2_reg=1e-4,
        bidirectional=True,
        output_activation="linear",
        use_layer_norm=False,
        num_heads=4,  # NEW: for multi-head attention
    ):
        super().__init__()

        kernel_regularizer = tf.keras.regularizers.l2(l2_reg)

        # Convolutional block
        self.conv = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation("relu")
        self.conv_dropout = tf.keras.layers.Dropout(dropout_rate)

        # GRU layer 1
        gru1 = tf.keras.layers.GRU(
            n_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            kernel_regularizer=kernel_regularizer,
        )
        self.gru_1 = tf.keras.layers.Bidirectional(gru1) if bidirectional else gru1

        # Multi-head attention
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=n_units,
            dropout=dropout_rate,
        )

        # GRU layer 2
        gru2 = tf.keras.layers.GRU(
            n_units,
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            kernel_regularizer=kernel_regularizer,
        )
        self.gru_2 = tf.keras.layers.Bidirectional(gru2) if bidirectional else gru2

        self.rnn_norm = tf.keras.layers.LayerNormalization() if use_layer_norm else tf.identity

        self.dense = tf.keras.layers.Dense(
            output_units,
            activation=output_activation,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.conv_dropout(x, training=training)

        # GRU 1
        x = self.gru_1(x, training=training)  # (batch, time, features)

        # Multi-head self-attention
        attention_out = self.attention(
            query=x, value=x, key=x, training=training
        )  # (batch, time, features)

        # Optional: residual connection (can improve training stability)
        x = x + attention_out

        # GRU 2
        x = self.gru_2(x, training=training)

        x = self.rnn_norm(x)
        return self.dense(x)


class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


def make_predictions(model, input_model, n_steps):
    X = input_model.copy()
    for step_ahead in range(n_steps):
        y_pred_one = model.predict(X[:, step_ahead:], verbose=0)
        y_pred_one = y_pred_one[:, :, np.newaxis]
        X = np.concatenate((X, y_pred_one), axis=1)

    return X[:, :, 0]


if __name__ == "__main__":
    num_series = 80
    series_size = 100
    n_steps = 16  # For SimpleNetRNN use 4

    input_serie = generate_series(num_series, series_size, incline=False)
    y_new = input_serie.copy()
    print(y_new.shape)
    scaler = DataScaler(y_new, n=None)
    y_new = scaler.rescale()
    size_ = int(0.8 * y_new.shape[0])
    x_train = y_new[:size_, :-n_steps]
    y_train = y_new[:size_, -n_steps:]
    x_test = y_new[size_:, :-n_steps]
    y_test = y_new[size_:, -n_steps:]

    print(x_train.shape, y_train.shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # model = SimpleNetRNN(n_neurons=10, output_units=n_steps)
    model = SimpleNetGRU(filters=10, kernel_size=4, n_units=5, strides=2, output_units=n_steps)
    # model(x_train[:, :, np.newaxis])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    history = model.fit(
        x_train[:, :, np.newaxis],
        y_train,
        epochs=100,
        validation_split=0.2,
        callbacks=[PrintDot(), early_stopping, reduce_lr],
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
    n_pred = 4
    X = make_predictions(model, x_train[:, :, np.newaxis], n_steps=n_pred)
    print(X.shape)
    X = scaler.scale(X)

    for i in range(5):
        plt.plot(range(len(input_serie[i, :])), input_serie[i, :], "o-", label="real value")
        plt.plot(
            range(len(X[i, :]))[-n_steps * n_pred :],
            X[i, :][-n_steps * n_pred :],
            "-r",
            label="prediction",
        )
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    # savel model
    model.save("./models/model_tensor", save_format="tf")
