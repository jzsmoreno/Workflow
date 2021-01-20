import streamlit as st
from pylab import bone, pcolor, colorbar, plot, show


from Soms_FraudDetection import somTrained, getData, transformData, getFrauds, getAccuracy

# Importacion del conjunto de datos
dataset, features, isFraud = getData()
# Preprocesamiento de los datos
features,sc = transformData(features)

models = {
    "Self-organizing map": somTrained(features)
}

# Sección de introducción
st.title("Predicción de fraudes usando mapas autoorganizados")
st.write(
    """
    * Bienvenid@ a este sencillo ejemplo que ejecuta un modelo entrenado 
    de IA usando mapas autoorganizados para encontrar potenciales fraudes.

    * La base de datos utilizada proviene del siguiente link: https://archive.ics.uci.edu/ml/datasets/credit+approval
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

st.sidebar.write(f"Configurar Distancia Interneuronal")

dist_int = st.sidebar.slider(
    "Distancia Interneuronal Recomendada : 0.9", 0.0, 1.0, 0.9, 0.1
)
 

# Sección de datos
st.write(
    """
    A continuación se muestran los primeros 5 registros de los datos utilizados:
    """
)
st.dataframe(dataset.head())


# Sección de datos
st.write(
    """
    * Mapa de colores de los clusters formados
    """
)

# Obtenemos la gráfica de colores
som = models["Self-organizing map"]
bone()
pcolor(som.distance_map().T)
colorbar()
for i, x in enumerate(features):
	w = som.winner(x)
	plot(w[0] + 0.5,
		w[1] + 0.5,
		'o',
		markeredgecolor = 'r',
		markerfacecolor = 'None',
		markersize = 10,
		markeredgewidth = 2)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot() 


# Obtención de los posibles fraudes
Possiblefrauds = getFrauds(som,features,dist_int,sc)


# Obtengo los CustomerID de los posibles fraudes
fraud_id = Possiblefrauds[:,0]


st.write("* Total de posibles fraudes : ")
st.header(str(len(fraud_id)))

st.dataframe(Possiblefrauds)


# Obtengo la precición del modelo
acc = getAccuracy(dataset,fraud_id)

st.write("* Porcentaje de predicción : ")
st.header(str(round(acc, 2))+'%')