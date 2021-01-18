import streamlit as st
import pandas as pd
import numpy as np
from likelihood import generate_series
from tensorflow.keras.models import load_model

# This files are in the forecasting folder
from series import forecasting, convert_to_df, create_train_test_set, rescale, scale
from figure import plot_series
from figure import plot_time_series
from figure import plot_displot

np.random.seed(0)
neural_network = load_model('forecasting/data_series.h5')

models = {
    "Neural Network": neural_network
}

# Sección de introducción
st.title("Predicción de series temporales usando Redes Neuronales")
st.write(
    """
    Bienvenid@ a este sencillo ejemplo que ejecuta un modelo entrenado 
    de IA usando redes neuronales recurrentes para predecir series de tiempo. 
    """
)

# Selección del modelo
model_selector = st.sidebar.selectbox(
    "Selecciona el modelo a utilizar:",
    list(models.keys())
)

# Especificación de datos
st.sidebar.markdown(
    """
    # Especificamos las características:
    """
)

# Or even better, call Streamlit functions inside a "with" block:

val = st.sidebar.radio('Inclinación', (False, True))
st.sidebar.write(f"You are in {val} slope value!")

n = st.sidebar.slider(
    "Número total de series", 2, 200, 25, 1
)

i_serie = st.sidebar.slider(
    "Serie a considerar", 0, n-1, 0, 1
)

#val = st.sidebar.slider(
#    "Inclinación", 0, 1, 0, 1
#)

p_train = st.sidebar.slider(
    "Proporción", 0.25, 1.0, 0.75, 0.05
)

# The dataset is generated
data_ = generate_series(n, n_steps = 100, incline = val)
data_scale, values = rescale(np.copy(data_))
series = convert_to_df(np.copy(data_)).describe()
train, test = create_train_test_set(np.copy(data_scale), p_train = p_train)

# Sección de datos
st.write(
    """
    A continuación se muestran las estadísticas descriptivas de los datos utilizados:
    """
)
st.dataframe(series)

# Sección de datos
st.write(
    """
    Y un pequeño gráfico generado con Plotly de una de las series temporales.
    """
)
fig1 = plot_time_series(convert_to_df(data_), i = i_serie)
st.plotly_chart(fig1)

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

st.write(
    """
    Ahora veamos la desviación de nuestra predicción: 
    """
)

st.write(
    """
    El error cuadrático medio del modelo es: 
    """
)
st.header(str(round(model.evaluate(train[0], train[1])[0], 4)))


fig3 = plot_displot(convert_to_df(data_), prediction, i = i_serie)
st.plotly_chart(fig3)

st.write(
    """
    La exactitud del modelo es: 
    """
)
st.header(str(round(model.evaluate(train[0], train[1])[1]*100))+'%')


