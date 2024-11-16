import numpy as np
import pandas as pd
import streamlit as st
from figure import plotly_figure_1  # Función para la visualización
from grasp import grasp  # Función para el algoritmo GRASP
from numpy import loadtxt
from tools import recombine_and_split

# Carga de los datos
data = pd.read_csv("heuristic/ubicaciones.csv")
pattern = "heuristic/evalu.txt_part_*"  # Ajustar a tus nombres de archivo
evalu = recombine_and_split(pattern)

# Sección de introducción
st.title("Optimización de Territorios Usando Heurística GRASP")
st.markdown(
    """
    **Bienvenid@ a la optimización de territorios y rutas de entrega utilizando el algoritmo GRASP.**

    Este modelo tiene como objetivo asignar clientes a territorios específicos optimizando la ruta de entrega, teniendo en cuenta el número de días de entrega disponibles y otras variables importantes.

    GRASP (Greedy Randomized Adaptive Search Procedure) es un algoritmo heurístico de optimización que ofrece soluciones cercanas al óptimo en problemas complejos como la asignación de territorios.
    """
)

# Sección de carga de datos
st.header("1. Datos de Entrada")
st.write(
    """
    A continuación se muestran los datos utilizados para la ejecución del modelo. Estos incluyen las ubicaciones de los clientes, sus características y los parámetros relevantes para la asignación de territorios.
    """
)
st.dataframe(data.head())  # Mostrar las primeras filas de los datos

# Barra lateral para configuración de parámetros
st.sidebar.header("2. Configuración de Parámetros")
st.sidebar.write(
    """
    Ajusta los parámetros del modelo para personalizar la asignación de territorios y rutas de entrega.
    """
)

# Selección del número de días de entrega
n_days = st.sidebar.slider(
    "Número de Días de Entrega",
    min_value=3,
    max_value=6,
    value=4,
    step=1,
    help="Número de días en los que se deben realizar las entregas. Ajusta este valor según tu necesidad.",
)

# Parámetros de configuración para el algoritmo GRASP
st.sidebar.subheader("Parámetros de GRASP")

threshold_1 = st.sidebar.slider(
    "Umbral 1 (threshold_1)",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Valor de umbral para el primer criterio de selección en GRASP.",
)

threshold_2 = st.sidebar.slider(
    "Umbral 2 (threshold_2)",
    min_value=0.1,
    max_value=1.0,
    value=0.9,
    step=0.1,
    help="Valor de umbral para el segundo criterio de selección en GRASP.",
)
# Ejecución del algoritmo GRASP para obtener la asignación de clientes
st.sidebar.markdown("### Ejecución del Algoritmo")
with st.spinner("Ejecutando el algoritmo GRASP..."):
    asig = grasp(
        data,
        evalu,
        n_days,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
    )  # Llamada a la función GRASP con los parámetros configurados
st.sidebar.success("¡Cálculo completado!")  # Mostrar mensaje al finalizar

# Sección de resultados
st.header("3. Resultados")
st.write(
    """
    Ahora que el modelo ha ejecutado el algoritmo GRASP, podemos ver cómo se han asignado los clientes a los territorios y qué rutas de entrega se sugieren. El siguiente gráfico interactivo muestra los resultados en un mapa, donde cada color representa un día de entrega.
    """
)

# Visualización con Plotly
fig = plotly_figure_1(data, asig)
st.plotly_chart(fig)  # Mostrar el gráfico interactivo

# Información adicional sobre los resultados
st.write(
    """
    ### Interpretación de los Resultados

    En el gráfico interactivo anterior, cada grupo de puntos representa un territorio asignado a un día de entrega específico. Puedes hacer zoom, mover y explorar los diferentes territorios asignados.

    - **Días de entrega:** Los colores en el mapa representan diferentes días de entrega.
    - **Clientes asignados:** Los clientes en cada territorio están agrupados para una entrega eficiente dentro del marco de días establecidos.
    """
)

# Mostrar la asignación de territorios en formato tabla
st.subheader("Asignación de Clientes a Territorios")
st.write(
    """
    La siguiente tabla muestra cómo se han asignado los clientes a sus respectivos territorios para cada día de entrega.
    """
)
# Aquí puedes mostrar la asignación de clientes con el resultado final
# Si 'asig' es un DataFrame, puedes mostrarlo directamente, sino procesarlo según sea necesario
st.dataframe(asig)  # Mostrar las asignaciones

# Añadir una sección de conclusión
st.header("4. Conclusión")
st.write(
    """
    El algoritmo GRASP ha permitido obtener una solución eficiente para la asignación de territorios y rutas de entrega. Dependiendo de los parámetros establecidos (número de días de entrega, umbrales y factores), el modelo puede ofrecer diferentes configuraciones para adaptarse a las necesidades de tu problema.

    ¡Esperamos que esta herramienta te sea útil para la optimización de tus operaciones logísticas!
    """
)
