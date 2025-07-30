## ¿Qué son las Redes Neuronales Recurrentes (RNN)?

Las **Redes Neuronales Recurrentes** son un tipo de arquitectura de redes neuronales diseñada específicamente para trabajar con **secuencias de datos**, como:

- Texto
- Series temporales
- Señales de audio o video

A diferencia de las redes convolucionales (CNN) y las redes feedforward, las RNN tienen un mecanismo de _memoria_ que les permite recordar información procesada en pasos anteriores. Es decir, **el estado interno** de la red se mantiene entre los elementos de la secuencia.

## Arquitectura básica de una RNN simple

La **RNN simple** funciona con la idea de un bucle temporal: el output en cada paso se convierte en parte del input para el siguiente paso. Su estructura es:

$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

Donde:

- $h_t$: estado oculto en el tiempo $t$
- $x_t$: entrada en el tiempo $t$
- $f$: función de activación (ej. $\tanh$ o ReLU)

### Ejemplo visual:

Matemáticamente, supóngase que se tienen $h$ unidades ocultas, tamaño de batch $n$, y el número de entradas es $d$. Entonces, la entrada es $x_{t} \in \mathbb{R}^{d\times n}$ y el estado oculto del paso anterior es $h_{t-1} \in \mathbb{R}^{h\times n}$.

$$
x_{t} = \left(\begin{matrix}
x^{1,1}_{t} & \cdots & x^{1,n}_{t} \\
\vdots & \ddots & \vdots \\
x^{d,1}_{t} & \cdots & x^{d,n}_{t}
\end{matrix}\right)
$$

donde $W_{hh} \in \mathbb{R}^{h\times h}$, $W_{xh} \in \mathbb{R}^{h\times d}$ y $b_h \in \mathbb{R}^{h\times 1}$, y el estado oculto al tiempo $t$ para $d = 1$ (caso de series temporales univariable) es calculado como le sigue:

$$
h_{t} =
f\left(
\left(\begin{matrix}
w^{1,1}_{hh} & \cdots & w^{1,h}_{hh} \\
\vdots & \ddots & \vdots \\
w^{h,1}_{hh} & \cdots & w^{h,h}_{hh}
\end{matrix}\right)
\left(\begin{matrix}
h^{1,1}_{t-1} & \cdots & h^{1,n}_{t-1} \\
\vdots & \ddots & \vdots \\
h^{h,1}_{t-1} & \cdots & h^{h,n}_{t-1}
\end{matrix}\right)
+
\left(\begin{matrix}
w^{1,1}_{xh} x^{1,1}_{t} & \cdots & w^{1,1}_{xh} x^{1,n}_{t} \\
\vdots & \ddots & \vdots \\
w^{h,1}_{xh} x^{1,1}_{t} & \cdots & w^{h,1}_{xh} x^{1,n}_{t}
\end{matrix}\right)
+
\left(\begin{matrix}
b^{1}_{h} \\
\vdots \\
b^{h}_{h}
\end{matrix}\right)
\right)
$$

## Limitaciones de las RNN simples

- **Problema de gradientes desaparecientes/estallantes**: No pueden manejar bien dependencias a largo plazo (ej. en un texto, no "recuerdan" lo que sucedió muchas palabras atrás).

## LSTM: Long Short-Term Memory

La **LSTM** es una versión mejorada de la RNN simple. Introduce mecanismos para controlar qué información se retiene y cuál se olvida.

### Componentes principales:

- **Puerta de olvido (Forget Gate)** → Decide qué información del estado previo debe desecharse.
- **Puerta de entrada (Input Gate)** → Decide nuevas celdas a almacenar en la memoria.
- **Celda de memoria (Cell State)** → Mantiene información por un largo tiempo.
- **Puerta de salida (Output Gate)** → Calcula el output basado en la celda actual.

### ⚙️ Ecuaciones resumidas:

1. $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
2. $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
3. $\tilde{C}_{t} = \tanh(W_C [h_{t-1}, x_t] + b_C)$
4. $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
5. $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
6. $h_t = o_t * \tanh(C_t)$

Donde $\sigma$ es la función sigmoide, y $*$ es multiplicación elemento a elemento.

### Ventajas:

- Maneja dependencias largas.
- Es más estable numéricamente que las RNN simples.

---

## GRU: Gated Recurrent Unit

La **GRU** también resuelve el problema de los gradientes, pero con una estructura ligeramente más sencilla que la LSTM. Combina las **puertas de olvido y entrada en una sola puerta (z)**, además de fusionar el estado oculto y la celda.

### 🔧 Componentes principales:

- **Puerta de actualización (Update Gate):** Decide qué información actualizar.
- **Puerta de restablecimiento (Reset Gate):** Controla cuánto del pasado afectará a la nueva entrada.

### 📘 Ecuaciones resumidas:

1. $z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$
2. $r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$
3. $\tilde{h}_{t}= \tanh(W_h [r_t * h_{t-1}, x_t] + b_h)$
4. $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$

### Ventajas:

- Más ligera que la LSTM.
- Menos parámetros → entrenamiento más rápido en algunos casos.

---

## Comparación rápida

| Modelo   | Memoria a largo plazo | Capacidad de modelado | Velocidad / Tamaño               |
| -------- | --------------------- | --------------------- | -------------------------------- |
| **RNN**  | ❌ Limitada           | Baja                  | ⚡ Rápida pero inestable         |
| **LSTM** | ✅ Alta               | Alta                  | 🐢 Más lenta, más precisa        |
| **GRU**  | ✅ Media a alta       | Media-Alta            | ⚡⚡ Un poco más rápida que LSTM |
