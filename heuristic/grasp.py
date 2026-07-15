import random
from typing import List, Union

import numpy as np
import pandas as pd
from numba import jit, njit, prange

# Definir constante como parámetros al inicio
THRESHOLD_1 = 0.7  # Umbral para distancia (70%)
THRESHOLD_2 = 0.9  # Umbral para distancia (90%)
VOLUME_FACTOR = 1.1  # Factor de volumen límite (110%)
CLIENT_LIMIT_FACTOR = 1.1  # Factor de límite de clientes (110%)


# Función para elegir centros de distribución
@jit()
def centros(
    X: Union[List, np.ndarray],
    evalu: List[List[float]],
    n_days: int,
    threshold_1: float = THRESHOLD_1,
    threshold_2: float = THRESHOLD_2,
) -> List[int]:
    """Select distribution centers for the GRASP construction phase.

    This function builds a list of center customer IDs by randomly choosing an initial
    customer with frequency equal to 1, then iteratively selecting subsequent centers
    based on distance thresholds.

    Parameters
    ----------
    X : `list` or `numpy.ndarray`
        Customer array-like structure where each element contains at least:
        `[Id_Cliente, Frecuencia, Vol_Entrega, lat, lon]`.
    evalu : `list`
        Distance/cost matrix where `evalu[i]` contains distances from center `i` to
        all candidates.
    n_days : `int`
        Number of delivery days (number of centers to select).
    threshold_1 : `float`, optional
        Threshold multiplier used to choose the second center.
    threshold_2 : `float`, optional
        Threshold multiplier used for subsequent centers.

    Returns
    -------
    list
        List of selected center IDs (1-based IDs as stored in `X[...][0]`).
    """
    C = []
    aux = random.choice(X)

    # Buscar un cliente con frecuencia igual a 1
    while aux[1] != 1:
        aux = random.choice(X)
    C.append(int(aux[0]))

    dist = evalu[int(C[0] - 1)]
    aux = np.random.choice(np.array(dist))

    # Elegir un centro según el umbral threshold_1
    while aux <= np.mean(np.array(dist)) * threshold_1:
        aux = np.random.choice(np.array(dist))
    indx = dist.index(aux)
    C.append(int(X[indx][0]))

    for i in range(n_days - 2):
        dist = np.array(dist)
        dist += np.array(evalu[int(C[i + 1] - 1)])
        dist = list(dist)

        aux = np.random.choice(np.array(dist))
        while aux <= np.mean(np.array(dist)) * threshold_2:
            aux = np.random.choice(np.array(dist))
        indx = dist.index(aux)
        C.append(int(X[indx][0]))

    return C


# Función para asignar clientes a los centros de distribución
def clientes(
    C: List[int],
    X: Union[List, np.ndarray],
    asig: List[List[int]],
    frec: List[int],
    data: pd.DataFrame,
    evalu: List[List[float]],
    n_days: int,
    c1: float = 10**9,
    volume_limit_factor: float = VOLUME_FACTOR,
    client_limit_factor: float = CLIENT_LIMIT_FACTOR,
) -> None:
    """Assign customers to selected centers respecting frequency and capacity limits.

    Parameters
    ----------
    C : `list`
        Selected centers IDs (1-based, matching customer IDs in `X`).
    X : `list` or `numpy.ndarray`
        Customer array-like structure where each element contains at least:
        `[Id_Cliente, Frecuencia, Vol_Entrega, lat, lon]`.
    asig : `list`
        Assignment matrix to be filled with 1s.
        Shape: `[n_days][len(X)]`, where `asig[i][j] == 1` means customer `j` assigned to day `i`.
    frec : `list`
        Mutable remaining delivery frequency per customer index.
    data : `pandas.DataFrame`
        Source customer data; must include columns `Frecuencia` and `Vol_Entrega`.
    evalu : `list`
        Distance/cost matrix used for selecting the next customer for each center.
    n_days : `int`
        Number of delivery days/centers.
    c1 : `float`, optional
        Large penalty value used in the selection process.
    volume_limit_factor : `float`, optional
        Multiplier applied to the average per-day volume limit.
    client_limit_factor : `float`, optional
        Multiplier applied to the average per-day customer limit.

    Returns
    -------
    None
        The function updates `asig` and `frec` in-place.
    """
    c2 = c1 * 10
    ntotal = len(X)
    climit = data.Frecuencia.sum() / n_days
    vmax = data.Frecuencia * data.Vol_Entrega
    vlim = vmax.sum() / n_days

    for i in prange(n_days):
        dist = evalu[C[i] - 1].copy()
        dist[C[i] - 1] = c1
        asig[i][C[i] - 1] = 1
        frec[C[i] - 1] -= 1

        maxv = X[C[i] - 1][2]
        j = 1
        f = []

        # Mejorar la búsqueda de clientes a asignar
        for k in prange(len(frec)):
            if frec[k] > 1:
                f.append(dist[k])

        # Asignar clientes de forma eficiente
        for k in range(len(f)):
            aux = min(f)
            ind = f.index(aux)
            indx = dist.index(aux)

            # Aseguramos que el cliente no sea asignado previamente
            while frec[indx] < 1:
                dist[indx] = c2
                f[ind] = c2
                aux = min(f)
                indx = dist.index(aux)
                ind = f.index(aux)

            dist[indx] = c1
            f[ind] = c2
            asig[i][indx] = 1
            frec[indx] -= 1
            maxv += X[indx][2]
            j += 1

        # Verificación de los límites de volumen y número de clientes
        while sum(frec) >= 1:
            aux = min(dist)
            indx = dist.index(aux)
            while frec[indx] < 1:
                dist[indx] = c2
                aux = min(dist)
                indx = dist.index(aux)

            dist[indx] = c1
            asig[i][indx] = 1
            frec[indx] -= 1
            maxv += X[indx][2]
            j += 1

            # Control de volumen y clientes
            if maxv >= vlim * volume_limit_factor:
                break
            if j >= climit * client_limit_factor:
                break


# Función principal del algoritmo GRASP
def grasp(
    data: pd.DataFrame,
    evalu: List[List[float]],
    n_days: int,
    threshold_1: float = THRESHOLD_1,
    threshold_2: float = THRESHOLD_2,
    volume_limit_factor: float = VOLUME_FACTOR,
    client_limit_factor: float = CLIENT_LIMIT_FACTOR,
) -> List[List[int]]:
    X = data[["Id_Cliente", "Frecuencia", "Vol_Entrega", "lat", "lon"]].to_numpy()
    ntotal = len(X)
    print("ntotal : ", ntotal)
    X = list(X)
    k = False

    while not k:
        asig = [[0] * ntotal for _ in prange(n_days)]
        frec = list(np.array(data.Frecuencia))

        a = centros(X, evalu, n_days, threshold_1, threshold_2)
        clientes(
            a,
            X,
            asig,
            frec,
            data,
            evalu,
            n_days,
            volume_limit_factor=volume_limit_factor,
            client_limit_factor=client_limit_factor,
        )

        # Verificar si todos los clientes han sido asignados
        j = sum(1 for i in frec if i == 0)
        if j == ntotal:  # Si todos los clientes están asignados
            k = True

    return asig
