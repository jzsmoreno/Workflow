import matplotlib.pyplot as plt
import streamlit as st
from pylab import bone, colorbar, pcolor, plot
from Soms_FraudDetection import getData, getFrauds, getMetrics, load_model, transformData

# Importación del conjunto de datos
dataset, features, isFraud = getData()
# Preprocesamiento de los datos
features, sc = transformData(features)

# Cargar el modelo SOM
models = {"Self-Organizing Map": load_model("./fraud_detection/som.p")}

# Configuración del layout
st.set_page_config(page_title="Detección de Fraudes", layout="wide")

# Sección de introducción
st.title("Predicción de Fraudes usando Mapas Auto-Organizados")
st.markdown(
    """
    **Bienvenid@ a la herramienta de detección de fraudes.**
    
    Esta aplicación utiliza Mapas Auto-Organizados (SOM) para predecir posibles fraudes en el conjunto de datos de aprobación de crédito. 
    A continuación, puedes ver cómo el modelo ha identificado los posibles fraudes, junto con métricas de precisión.
    
    #### Dataset utilizado:
    - [UCI Credit Approval Dataset](https://archive.ics.uci.edu/ml/datasets/credit+approval)
    """
)

# Panel lateral para selección de modelo y parámetros
st.sidebar.markdown("## Configuración del Modelo")

# Selección del modelo
model_selector = st.sidebar.selectbox("Selecciona el modelo a utilizar:", list(models.keys()))

# Especificación de parámetros
dist_int = st.sidebar.slider("Distancia Interneuronal Recomendada:", 0.0, 1.0, 0.9, 0.1)

# Mostrar los primeros 5 registros de los datos
st.markdown("### Primeros 5 registros de los datos utilizados:")
st.dataframe(dataset.head(), use_container_width=True)

# Crear columnas para distribuir los gráficos y resultados
col1, col2 = st.columns([2, 1])

# Gráfico SOM
with col1:
    st.markdown("### Mapa de Distancias de los Clusters Formados:")

    som = models["Self-Organizing Map"]

    # Crear una figura explícita
    fig, ax = plt.subplots(figsize=(10, 8))  # Tamaño ajustable
    bone()
    pcolor(som.distance_map().T)
    colorbar()

    # Agregar los puntos correspondientes al ganador de cada característica
    for i, x in enumerate(features):
        w = som.winner(x)
        ax.plot(
            w[0] + 0.5,
            w[1] + 0.5,
            "o",
            markeredgecolor="r",
            markerfacecolor="None",
            markersize=10,
            markeredgewidth=2,
        )

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig, use_container_width=True)

# Resultados de predicción en la columna derecha
with col2:
    st.markdown("### Resultados de Predicción:")

    Possiblefrauds = getFrauds(som, features, dist_int, sc)

    # Extraer los ID de los clientes potencialmente fraudulentos
    fraud_id = Possiblefrauds[:, 0]
    acc = getMetrics(dataset, fraud_id)

    # Verificar si acc es una lista y obtener el primer valor si es necesario
    if isinstance(acc, list):
        acc = acc[0]  # Obtén el primer valor si acc es una lista

    # Mostrar las métricas
    st.write(f"**Total de posibles fraudes detectados:** {len(fraud_id)}")
    st.write(f"**Porcentaje de precisión del modelo:** {round(acc, 2)}%")

    # Mostrar los datos de los fraudes detectados
    st.markdown("### Detalles de los posibles fraudes:")
    st.dataframe(Possiblefrauds, use_container_width=True)

    # Puedes agregar más información aquí, por ejemplo:
    st.markdown("#### Análisis de los principales fraudes:")
    st.write("Visualización detallada de los clientes que podrían ser fraudulentos en el gráfico.")
    # Agregar otras métricas o análisis adicionales

# Opcional: Agregar una sección final con explicaciones
st.markdown(
    """
    ## ¿Cómo Funciona el Modelo?
    
    El modelo utiliza Mapas Auto-Organizados (SOM) para agrupar datos similares y detectar anomalías que podrían indicar fraudes.
    Estos modelos son útiles para análisis no supervisados donde no se dispone de etiquetas claras sobre los fraudes.
    
    **Nota**: La precisión depende de los parámetros utilizados en el SOM. Ajusta la *Distancia Interneuronal* para observar cambios en el resultado.
    """
)
