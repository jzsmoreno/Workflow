# Explicación del Algoritmo GRASP para Asignación de Clientes a Centros de Distribución

## Introducción

El algoritmo **GRASP** (Greedy Randomized Adaptive Search Procedure) es una técnica de optimización heurística utilizada para resolver problemas combinatorios complejos. En este caso, el objetivo es asignar clientes a centros de distribución, considerando diversos factores como las distancias entre clientes y centros, las frecuencias de entrega y los límites de volumen y clientes.

Este enfoque se divide en dos fases:
1. **Construcción aleatoria greedy**: Se seleccionan centros de distribución de manera aleatoria pero guiada por un criterio de selección local.
2. **Búsqueda local**: Se ajustan las asignaciones para optimizar el cumplimiento de los límites de volumen y de clientes asignados a cada centro.

## Descripción de las Funciones

### 1. Selección de Centros de Distribución (Función `centros`)

Esta función se encarga de seleccionar los centros de distribución en función de las distancias entre los clientes y los centros disponibles. El proceso de selección se realiza en base a los siguientes pasos:

- **Selección inicial aleatoria**: Se selecciona un cliente aleatorio que tenga una frecuencia de entrega igual a 1.
- **Selección de centros subsecuentes**: Para cada nuevo centro, se evalúan las distancias de todos los clientes a los centros ya seleccionados. Se elige un cliente cuya distancia a los centros previos sea mayor a un umbral específico (controlado por `threshold_1` para el primer centro y `threshold_2` para los siguientes).
- **Repetición hasta completar el número de días**: Se repite el proceso de selección hasta que se hayan elegido los centros necesarios para el número de días (`n_days`).

### 2. Asignación de Clientes a Centros de Distribución (Función `asignar_clientes`)

Esta función es responsable de asignar a los clientes a los centros de distribución seleccionados. El proceso se realiza de la siguiente manera:

- **Inicialización de variables**: Se inicializan las matrices de asignación y las frecuencias de los clientes.
- **Asignación greedy**: Para cada día, se calcula la distancia de todos los clientes a cada centro. Se asigna el cliente más cercano, respetando el límite de clientes y el volumen de entrega.
- **Optimización de la asignación**: Después de asignar el primer cliente, se verifica si los centros están alcanzando el límite de volumen y número de clientes. Si es necesario, se ajusta la asignación para asegurar que los centros no excedan estos límites.
- **Control de límites**: Se asegura de que los centros no excedan los límites de volumen de entrega y el número de clientes asignados a cada centro.

### 3. Función Principal GRASP (Función `grasp`)

La función principal implementa el flujo del algoritmo GRASP, ejecutando las fases de construcción y búsqueda local iterativamente:

- **Entrada de Datos**: La función recibe un conjunto de datos con información sobre los clientes (ID, frecuencia, volumen de entrega, ubicación geográfica).
- **Construcción de Centros**: Llama a la función `centros` para seleccionar los centros de distribución.
- **Asignación de Clientes**: Llama a la función `asignar_clientes` para asignar los clientes a los centros seleccionados.
- **Repetición**: El proceso se repite hasta que todos los clientes sean asignados, es decir, cuando todas las frecuencias de los clientes se han agotado.

## Parámetros de Entrada

- **`data`**: Un `DataFrame` con la información de los clientes, incluyendo el ID del cliente, la frecuencia de entrega, el volumen de entrega y la ubicación geográfica (latitud y longitud).
- **`evalu`**: Una matriz que representa las distancias entre los centros de distribución.
- **`n_days`**: El número de días a asignar clientes a centros de distribución.
- **`threshold_1` y `threshold_2`**: Umbrales de selección para elegir los centros en las fases de construcción. Controlan la "distancia mínima" para elegir el siguiente centro.
- **`volume_limit_factor`**: Un factor multiplicador que ajusta el límite de volumen que puede manejar cada centro de distribución.
- **`client_limit_factor`**: Un factor multiplicador que ajusta el número máximo de clientes que puede manejar cada centro de distribución.

## Flujo de Ejecución

1. **Selección inicial de centros**: El algoritmo selecciona un centro de distribución aleatoriamente y, a partir de él, selecciona más centros según las distancias entre ellos y los umbrales establecidos.
2. **Asignación de clientes**: A continuación, asigna clientes a los centros de distribución, teniendo en cuenta las distancias y los límites de volumen y clientes.
3. **Repetición de procesos**: El proceso de selección y asignación se repite hasta que todos los clientes estén asignados a un centro de distribución.
