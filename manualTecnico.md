# ia-pr2

# Manual Técnico de Modelos de Machine Learning

Este manual técnico describe los modelos de machine learning implementados, incluyendo su configuración, estructuras de datos de entrada, parámetros utilizados, y una explicación detallada de su funcionamiento.

## Índice

1. [Introducción](#introducción)
2. [Estructura de Datos](#estructura-de-datos)
3. [Modelos Implementados](#modelos-implementados)
   - [Regresión Lineal](#regresión-lineal)
   - [Regresión Polinómica](#regresión-polinómica)
   - [Árbol de Decisión](#árbol-de-decisión)
   - [Red Neuronal](#red-neuronal)
   - [K-Means](#k-means)
   - [K-Nearest Neighbor](#k-nearest-neighbor)
4. [Parseo del CSV](#parseo-del-csv)
5. [Errores Comunes y Soluciones](#errores-comunes-y-soluciones)
6. [Recomendaciones](#recomendaciones)

---

## Introducción

Este documento detalla el funcionamiento de una serie de modelos de machine learning implementados en un sistema. Cada modelo se elige según el tipo de problema que se desea resolver, y el procesamiento de datos está adaptado para cada uno. Los modelos permiten realizar predicciones y clasificaciones basadas en diferentes algoritmos de machine learning, tales como regresión, clustering y clasificación.

## Estructura de Datos

Para todos los modelos, los datos se cargan mediante un archivo CSV que se analiza y procesa antes de ser utilizados. El CSV está estructurado de la siguiente manera:

```plaintext
Entrenamiento;Punto;Euclideano;Manhattan
"A,9,8,8,I;B,2,1,2,III;C,9,8,9,I;D,8,9,9,II;E,1,8,2,I;F,9,8,8,III;G,2,2,3,II";"H,5,7,9";"II";"I,II"
```

Cada campo representa lo siguiente:

<ul>
    <li>
        <b>Entrenamiento:</b> Datos de entrenamiento separados por comas.
    </li>
    <li>
        <b>Punto:</b> Punto específico para clasificar o predecir.
    </li>
    <li>
        <b>Euclideano:</b> Tipo de métrica o configuración para el cálculo.
    </li>
    <li>
        <b>Manhattan:</b> Otra métrica o configuración.
    </li>
</ul>


## Modelos Implementados

### Regresión Lineal

Descripción
El modelo de regresión lineal encuentra una relación lineal entre una variable dependiente y una o más variables independientes.

Estructura de Datos

<ul>
    <li>
        <b>Entrada</b>: Dos columnas de datos, x (variable independiente) y y (variable dependiente).
    </li>
    <li>
        <b>Salida</b>: Coeficientes de la recta de mejor ajuste.
    </li>
</ul>

Parámetros

<ul>
    <li>
        <b>xTrain</b>: Valores de x utilizados para el entrenamiento.
    </li>
    <li>
        <b>yTrain</b>: Valores de y correspondientes para el entrenamiento.
    </li>
</ul>

Código de Implementación

```js
if (modelType == "linear-regression") {
    lines.forEach(line => {
        const [x, y] = line.split(';').map(parseFloat);
        xTrain.push(x);
        yTrain.push(y);
    });
}
```

### Regresión Polinómica

Descripción

La regresión polinómica es una extensión de la regresión lineal que permite modelar relaciones de orden superior entre las variables.

Estructura de Datos

<ul>
    <li>
        <b>Entrada</b>: Columnas de x, y, y xPred.
    </li>
    <li>
        <b>Salida</b>: Coeficientes del polinomio de ajuste.
    </li>
</ul>

Parámetros
<ul>
    <li>
        <b>xTrain</b>: Valores de x para el entrenamiento.
    </li>
    <li>
        <b>yTrain</b>: Valores de y para el entrenamiento.
    </li>
    <li>
        <b>xToPredict</b>: Valores de x sobre los que se harán predicciones.
    </li>
</ul>

Código de Implementación

```js
if (modelType == "polynomial-regression") {
    lines.forEach(line => {
        const [x, y, xPred] = line.split(';').map(parseFloat);
        xTrain.push(x);
        yTrain.push(y);
        xToPredict.push(xPred);
    });
}
```

### Árbol de Decisión

Descripción

El modelo de árbol de decisión realiza clasificaciones y predicciones mediante un árbol jerárquico de decisiones, basado en la estructura de los datos.

Estructura de Datos

<ul>
    <li>
        <b>Entrada</b>: Varias columnas representando atributos y etiquetas de clase.
    </li>
    <li>
        <b>Salida</b>: Estructura de árbol para realizar clasificaciones.
    </li>
</ul>

Parámetros
<ul>
    <li>
        <b>headers</b>: Nombres de los atributos en la primera fila del CSV.
    </li>
    <li>
        <b>xTrain</b>: Datos de entrenamiento, incluyendo atributos y etiquetas de clase.
    </li>
</ul>

Código de Implementación

```js
if (modelType === "decision-tree") {
    headers = lines[0].split(','); // Cabeceras de atributos
    const trainData = [];
    lines.slice(1).forEach(line => {
        const attributes = line.split(',').map(item => item.trim());
        trainData.push(attributes);
    });
    xTrain = [headers, ...trainData];
}
```

### Red Neuronal

Descripción

La red neuronal es un modelo de machine learning inspirado en el cerebro humano, adecuado para problemas complejos y de gran dimensión.

Estructura de Datos

<ul>
    <li>
        <b>Entrada</b>: Valores numéricos.
    </li>
    <li>
        <b>Salida</b>: Predicciones o clasificaciones basadas en las capas de la red.
    </li>
</ul>

Parámetros
<ul>
    <li>
        <b>arrNR</b>: Array de valores numéricos.
    </li>
</ul>

Código de Implementación

```js
if (modelType == "neuronal-network") {
    arrNR = lines[0].split(',').map(Number); // Convierte cada valor a entero
}
```

### K-Means

Descripción

K-means es un algoritmo de clustering que agrupa los datos en k clusters basándose en la distancia euclidiana.

Estructura de Datos

<ul>
    <li>
        <b>Entrada</b>: Número de clusters, datos de entrenamiento, número de iteraciones.
    </li>
    <li>
        <b>Salida</b>: Clusters con datos agrupados.
    </li>
</ul>

Parámetros
<ul>
    <li>
        <b>clusters</b>: Número de clusters especificado.
    </li>
    <li>
        <b>trainingData</b>: Datos de entrenamiento para el clustering.
    </li>
    <li>
        <b>iterations</b>: Número de iteraciones.
    </li>
</ul>

Código de Implementación

```js
if (modelType === "k-means-linear") {
    lines.forEach(line => {
        const [numeroClusters, entrenamiento, numeroIteraciones] = line.split(';');
        clusters = parseInt(numeroClusters, 10);
        trainingData = entrenamiento.split(',').map(Number);
        iterations = parseInt(numeroIteraciones, 10);
    });
}
```

### K-Nearest Neighbor

Descripción

K-Nearest Neighbor (KNN) es un algoritmo de clasificación que clasifica nuevos datos en función de los k puntos de datos más cercanos.

Estructura de Datos

<ul>
    <li>
        <b>Entrada</b>:  Datos de entrenamiento, punto de predicción, métricas.
    </li>
    <li>
        <b>Salida</b>: Clasificación o predicción basada en los vecinos cercanos.
    </li>
</ul>

Parámetros
<ul>
    <li>
        <b>entrenamientoValue</b>: Datos de entrenamiento en formato CSV.
    </li>
    <li>
        <b>puntoValue</b>: Punto a clasificar.
    </li>
    <li>
        <b>euclideanoValue</b>: Métrica euclidiana para KNN.
    </li>
    <li>
        <b>manhattanValue</b>: Métrica Manhattan para KNN.
    </li>
</ul>

Código de Implementación

```js
if (modelType === "k-nearest-neighbor") {
    const [entrenamiento, punto, euclideano, manhattan] = lines[0].split(';').map(item => item.trim());
    const entrenamientoValue = entrenamiento.replace(/^"|"$/g, '');
    const puntoValue = punto.replace(/^"|"$/g, '');
    const euclideanoValue = euclideano.replace(/^"|"$/g, '');
    const manhattanValue = manhattan.replace(/^"|"$/g, '');
}
```
## Parseo del CSV

La función de parseo permite cargar y procesar los datos en función del tipo de modelo. La función se asegura de extraer correctamente cada valor del CSV y prepararlo para su uso en el modelo correspondiente.

```json 
function parseCSVData(data) {
    // Lógica de parseo según el tipo de modelo
}
```

### Errores Comunes y Soluciones

<ol>
    <li>
        <b>Datos en Formato Incorrecto</b>: Verificar el formato del CSV, ya que un error común es la falta de comillas en los valores.
    </li>
    <li>
        <b>Número de Parámetros Insuficiente</b>: Cada modelo requiere un conjunto específico de datos, asegúrate de que cada campo esté correctamente poblado.
    </li>
    <li>
        <b>Error de Tipo de Dato</b>: Usar parseFloat o parseInt cuando sea necesario para convertir datos numéricos.
    </li>
</ol>

### Recomendaciones

<ol>
    <li>
        <b>Validación de Datos</b>: Implementa funciones de validación antes de procesar el CSV.
    </li>
    <li>
        <b>Logging</b>: Utiliza console.log o herramientas de logging para verificar que los datos se procesen correctamente.
    </li>
    <li>
        <b>Documentación Continua</b>:  Actualiza este manual a medida que se agreguen o modifiquen modelos.
    </li>
</ol>