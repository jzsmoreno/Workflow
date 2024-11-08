"""
Created on Tue Jan 12 18:44:38 2021

@author: J. Ivan Avalos
"""

import pickle

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


#################### Importamos el data set ####################
def getData():
    dataset = pd.read_csv("fraud_detection/data/Credit_Card_Applications.csv")
    # Obtenemos las caracteristicas
    features = dataset.iloc[:, :-1].values
    # Obtenemos las etiquetas
    isFraud = dataset.iloc[:, -1].values
    return dataset, features, isFraud


#################### Preprocesamiento de los datos ####################
def transformData(features):
    sc = MinMaxScaler(feature_range=(0, 1))
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


def somTrained(features, x=10, y=10, sigma=0.5, learning_rate=0.01, num_iteration=1000):
    num_features = features.shape[1]
    som = MiniSom(x=x, y=y, input_len=num_features, sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(features)
    som.train_random(data=features, num_iteration=num_iteration)
    return som


def getFrauds(som, features, dist_int, sc):
    # Obtención los clusters
    mappings = som.win_map(features)

    # Obtengo los indices de los clusters
    distance_map = som.distance_map().round(2)
    n = distance_map.shape[0]
    bestIdx = [[i, j] for i in range(n) for j in range(n) if (distance_map[i, j] >= dist_int)]

    # Obtengo los potenciales fraudes
    fraud_list = []  # Arreglo de numpys con los posibles fraudes
    num_frauds = 0
    for x in bestIdx:
        fraud_list.append(mappings[(x[0], x[1])])

    Possiblefrauds = []
    for frauds in fraud_list:
        for fraud in frauds:
            Possiblefrauds.append(fraud)

    fraud_inverse_transformed = sc.inverse_transform(Possiblefrauds)

    return fraud_inverse_transformed


def getMetrics(dataset, fraud_id):
    # Variables to keep track of the number of correct and total predictions
    true_positives = 0  # Correctly predicted frauds
    true_negatives = 0  # Correctly predicted non-frauds
    false_positives = 0  # Non-frauds predicted as frauds
    false_negatives = 0  # Frauds predicted as non-frauds
    total_predictions = len(dataset)

    for index, customer_id in enumerate(dataset["CustomerID"]):
        actual_class = dataset["Class"][index]

        # Check if the current customer is a fraud
        is_fraud = customer_id in fraud_id

        # Update confusion matrix counts
        if actual_class == 1 and is_fraud:  # True positive
            true_positives += 1
        elif actual_class == 0 and not is_fraud:  # True negative
            true_negatives += 1
        elif actual_class == 0 and is_fraud:  # False positive
            false_positives += 1
        elif actual_class == 1 and not is_fraud:  # False negative
            false_negatives += 1

    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / total_predictions * 100

    # Calculate precision
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives) * 100
    else:
        precision = 0  # Avoid division by zero

    # Calculate recall
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives) * 100
    else:
        recall = 0  # Avoid division by zero

    # Calculate F1-Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0  # Avoid division by zero

    # Output the metrics
    print("MinSom accuracy : ", accuracy)
    print("MinSom precision : ", precision)
    print("MinSom recall : ", recall)
    print("MinSom F1-score : ", f1_score)

    return accuracy


def load_model(filepath):
    # Load the trained model from the file
    model = pickle.load(open(filepath, "rb"))
    return model


if __name__ == "__main__":
    # Cargar datos
    dataset, features, isFraud = getData()
    features_transformed, sc = transformData(features)
    # Obtener los clusters
    som = somTrained(features_transformed, 3, 3, 1)
    # Obtener los posibles fraudes
    fraud_id = getFrauds(som, features_transformed, 0.75, sc)
    # Obtener la precisión
    metrics = getMetrics(dataset, fraud_id)
    filepath = "./fraud_detection/som.p"
    with open(filepath, "wb") as outfile:
        pickle.dump(som, outfile)

    som = load_model(filepath)
