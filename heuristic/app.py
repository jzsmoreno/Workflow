import numpy as np
import pandas as pd
import streamlit as st
from figure import plotly_figure_1  # Función para la visualización
from grasp import grasp  # Función para el algoritmo GRASP
from numpy import loadtxt
from tools import recombine_and_split

# Carga de los datos
data = pd.read_csv("heuristic/ubicaciones.csv")
pattern = "heuristic/evalu.txt_part_*"  # Example pattern, adjust to your filenames
evalu = recombine_and_split(pattern)

# evalu = loadtxt("heuristic/evalu.txt").tolist()

# Sección de introducción
st.title("Optimización de Territorios Usando Heurística GRASP")
st.write(
    """
    Bienvenid@ a este ejemplo que ejecuta un modelo GRASP para la optimización de territorios y la asignación de rutas de entrega.
    """
)

# Sección de datos
st.write(
    """
    A continuación se muestran los datos utilizados en el modelo:
    """
)
st.dataframe(data.head())  # Mostrar las primeras filas de los datos

# Sección de configuración a través de la barra lateral
st.sidebar.markdown(
    """
    # Configuración de los Parámetros:
    """
)

# Selección del número de días de entrega a través de un slider en la barra lateral
n_days = st.sidebar.slider("Número de Días de Entrega", min_value=3, max_value=6, value=4, step=1)

# Ejecución del algoritmo GRASP para obtener la asignación de clientes
# Mostramos un spinner mientras se calcula la asignación
with st.spinner("Ejecutando el algoritmo GRASP..."):
    asig = grasp(data, evalu, n_days)
st.success("¡Cálculo completado!")  # Mostrar mensaje al finalizar

# Sección de visualización de los resultados
st.write(
    """
    Veamos los clusters de clientes divididos por Día de Entrega en el mapa interactivo:
    """
)

# Visualización con Plotly
fig = plotly_figure_1(data, asig)
st.plotly_chart(fig)  # Mostrar el gráfico interactivo
