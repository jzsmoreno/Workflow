import numpy as np
import pandas as pd
import streamlit as st
from figure import plotly_figure_1

# This files are in the src folder
from grasp import grasp
from numpy import loadtxt

data = pd.read_csv("heuristic/ubicaciones.csv")
evalu = loadtxt("heuristic/evalu.txt")
evalu = evalu.tolist()


# Sección de introducción
st.title("Optimización de Territorios Usando Heurística GRASP")
st.write(
    """
    Bienvenid@ a este ejemplo que ejecuta un modelo Grasp. 
    """
)

# Sección de datos
st.write(
    """
    A continuación los datos utilizados.
    """
)
st.dataframe(data.head())
# Especificación de datos
st.sidebar.markdown(
    """
    # Especificamos los Dias de Entrega:
    """
)
n_days = st.sidebar.slider("Número de Días de Entrega", 3, 6, 4, 1)
asig = grasp(data, evalu, n_days)
# Sección de datos
st.write(
    """
    Veamos los clusters de clientes divididos por Día de Entrega.
    """
)
fig = plotly_figure_1(data, asig)
st.plotly_chart(fig)
