"""
Created on Tue Jan 12 18:44:38 2021

@author: J. Ivan Avalos
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

#################### Importamos el data set ####################
def getData():
    dataset = pd.read_csv('fraud_detection/data/Credit_Card_Applications.csv')
    # Obtenemos las caracteristicas
    features = dataset.iloc[:, :-1].values
    # Obtenemos las etiquetas
    isFraud = dataset.iloc[:, -1].values
    return dataset,features, isFraud

#################### Preprocesamiento de los datos ####################
def transformData(features):
    sc = MinMaxScaler(feature_range = (0, 1))
    # Transformamos los datos en el rango [0,1] 
    features = sc.fit_transform(features)
    return features, sc 


#################### Entrenamiento SOM ####################
"""
- x, y son las dimensiones del SOM:
    ejemplo: si x = 10, y = 10, entonces se formaran 100 clusters
- input_len es la dimension de las instancias
- sigma es el radio de barrido
- learning_rate es la tasa de aprendizaje, indica que tan rapido se disminuye el radio
"""
def somTrained(features, x=10, y=10, sigma=1.0, learning_rate=0.3, num_iteration=100):
    num_features = features.shape[1]
    som = MiniSom(x = x, y = y, input_len = num_features, sigma = sigma, learning_rate = learning_rate)
    som.random_weights_init(features)
    som.train_random(data = features, num_iteration = num_iteration)
    return som

def getFrauds(som,features,dist_int,sc):
    # Obtención los clusters 
    mappings = som.win_map(features)

    # Obtengo los indices de los clusters
    distance_map = som.distance_map().round(1)
    bestIdx = [[i,j] for i in range(10) for j in range(10) if(distance_map[i,j]>=dist_int)]

    # Obtengo los potenciales fraudes
    fraud_list = [] # Arreglo de numpys con los posibles fraudes
    num_frauds = 0
    for x in bestIdx:
        fraud_list.append(mappings[(x[0],x[1])])

    Possiblefrauds = []
    for frauds in fraud_list:
        for fraud in frauds:
            Possiblefrauds.append(fraud)

    fraud_inverse_transformed = sc.inverse_transform(Possiblefrauds)

    return fraud_inverse_transformed

def getAccuracy(dataset,fraud_id):
    right_prediction_index =[]
    wrong_prediction_index =[]
    for fraudsbySom in fraud_id:
        for index,fraudsTrue in enumerate(dataset['CustomerID']):
            if(fraudsbySom == fraudsTrue):
                if(dataset['Class'][index] == 0):
                    right_prediction_index.append(index)
            else:
                wrong_prediction_index.append(index)
    return (len(right_prediction_index)/len(fraud_id))*100