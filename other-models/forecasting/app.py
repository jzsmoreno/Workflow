from sklearn.datasets import load_iris
import streamlit as st
import pandas as pd
import numpy as np
from likelihood import generate_series, rescale
from tensorflow.keras.models import load_model

# This files are in the src-forecasting folder
from series import forecasting, convert_to_df, create_train_test_set
from figure import plot_series
from figure import plot_time_series

np.random.seed(0)
neural_network = load_model('data_series.h5')

models = {
    "Neural Network": neural_network
}

# Sección de introducción
st.title("Predicción de series temporales usando Redes Neuronales")
st.write(
    """
    Bienvenid@ a este sencillo ejemplo que ejecuta un modelo entrenado 
    de AI. 
    """
)

# Especificación de datos
st.write(
    """
    Especificamos las características:
    """
)
n = st.slider(
    "Número total de series", 1, 100, 25, 1
)

i_serie = st.slider(
    "Serie a considerar", 0, n, 0, 1
)
p_train = st.slider(
    "Proporción", 0.5, 1.0, 0.75, 0.05
)

# The dataset is generated
data_ = generate_series(n, n_steps = 100)
data_scale, values = rescale(np.copy(data_))
series = convert_to_df(np.copy(data_)).describe()
train, test = create_train_test_set(np.copy(data_scale), p_train = p_train)

# Sección de datos
st.write(
    """
    A continuación los datos utilizados.
    """
)
st.dataframe(series)

# Sección de datos
st.write(
    """
    Y un pequeño gráfico generado con Plotly.
    """
)
fig1 = plot_time_series(convert_to_df(data_), i = i_serie)
st.plotly_chart(fig1)

# Selección del modelo
model_selector = st.sidebar.selectbox(
    "Selecciona el modelo a utilizar:",
    list(models.keys())
)
model = models[model_selector]

# Predicción

st.write(
    """
    De acuerdo a la predicción del modelo, el forecasting correspondiente es: 
    """
)

prediction = forecasting(model, train, data_.shape[1], values)

fig2 = plot_series(convert_to_df(data_), prediction, i = i_serie)
st.plotly_chart(fig2)
