import numpy as np
import pandas as pd
import streamlit as st
from figure import *
from likelihood.tools import *
from series import *
from tensorflow.keras.models import load_model

np.random.seed(0)
neural_network = load_model("forecasting/models/model_tensor")
neural_network.compile(loss="mse", optimizer="adam", metrics=["mae"])

models = {"Neural Network": neural_network}

# Sección de introducción
st.title("Predicción de Series Temporales usando Redes Neuronales")
st.write(
    """
    Bienvenid@ a este sencillo ejemplo que ejecuta un modelo entrenado 
    de IA usando redes neuronales recurrentes para predecir series de tiempo. 
    El modelo ha sido entrenado para hacer pronósticos precisos, basándose en datos históricos.
    """
)

# Selector de modelo en el sidebar
model_selector = st.sidebar.selectbox("Selecciona el modelo a utilizar:", list(models.keys()))

# Especificación de datos en el sidebar
st.sidebar.markdown("# Parámetros de Entrada:")
val = st.sidebar.radio("Inclinación", (False, True))
n = st.sidebar.slider("Número total de series", 2, 200, 25, 1)
i_serie = st.sidebar.slider("Serie a considerar", 0, n - 1, 0, 1)
p_train = st.sidebar.slider("Proporción de entrenamiento", 0.2, 1.0, 0.7, 0.1)
steps = st.sidebar.slider("Número de pasos del modelo", 1, 20, 16, 1)

# Generación de los datos
n_steps = 100
data_ = generate_series(n, n_steps=n_steps, incline=val)
scaler = DataScaler(np.copy(data_))
data_scale = scaler.rescale()
series = convert_to_df(np.copy(data_)).describe()

train, test = create_train_test_set(np.copy(data_scale), p_train=p_train, steps=steps)
neural_network.fit(train[0][:, :, np.newaxis], train[1], verbose=0)

# Sección de estadísticas descriptivas de los datos
st.header("Estadísticas Descriptivas de los Datos")
st.write("Aquí puedes ver las estadísticas de las series temporales generadas:")
st.dataframe(series)

# Sección de visualización de la serie temporal
st.header("Gráfico de la Serie Temporal Seleccionada")
st.write("A continuación, se muestra un gráfico de la serie temporal seleccionada:")
fig1 = plot_time_series(convert_to_df(data_), i=i_serie)
st.plotly_chart(fig1)

# Selección del modelo y predicción
model = models[model_selector]
st.header("Predicción del Modelo")
st.write("La siguiente es la predicción generada por el modelo para la serie seleccionada:")
prediction = forecasting(model, train[0][:, :, np.newaxis], data_.shape[1], scaler, steps=steps)

fig2 = plot_series(convert_to_df(data_), prediction, i=i_serie)
st.plotly_chart(fig2)

# Error cuadrático medio (RMSE)
st.header("Evaluación del Modelo")
st.write("A continuación se muestra el error cuadrático medio de la predicción:")
rmse = model.evaluate(train[0][:, :, np.newaxis], train[1])[0]
st.write(f"Error Cuadrático Medio (RMSE): {round(rmse, 4)}")

# Gráfico de la desviación
st.header("Desviación de la Predicción")
st.write("Aquí mostramos cómo la predicción difiere de la serie temporal original:")
fig3 = plot_displot(convert_to_df(data_), prediction, i=i_serie)
st.plotly_chart(fig3)

# Footer con más detalles
st.markdown("---")
st.write("Desarrollado por jzsmoreno. Para más información, visita nuestra documentación.")
