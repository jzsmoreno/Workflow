import streamlit as st
import pandas as pd
import numpy as np
from numpy import loadtxt
from numba import jit

# This files are in the src folder
from grasp import grasp
from figure import plotly_figure_1

data=pd.read_csv('heuristic/ubicaciones.csv')
evalu=loadtxt('heuristic/evalu.txt')
evalu=evalu.tolist()


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
    Especificamos los Dias de Entrega:
    """
)
n_days = st.sidebar.slider(
    "Número de Días de Entrega", 3.0, 4.0, 6.0, 1.0
)
asig=grasp(data,evalu,int(n_days))
# Sección de datos
st.write(
    """
    Veamos los clusters de clientes divididos por Día de Entrega.
    """
)
fig = plotly_figure_1(data,asig)
st.plotly_chart(fig)