# ia-pr2

# Manual de Usuario - Modelos de Machine Learning

Este manual proporciona instrucciones detalladas para utilizar el sistema de modelos de machine learning. Aquí encontrarás todo lo necesario para cargar datos, ejecutar modelos y obtener resultados.

## Índice

1. [Introducción](#introducción)
2. [Preparación de Datos](#preparación-de-datos)
3. [Modelos Disponibles](#modelos-disponibles)
4. [Formato del CSV](#formato-del-csv)
5. [Uso de los Modelos](#uso-de-los-modelos)
6. [Solución de Problemas](#solución-de-problemas)

## Introducción

Este sistema te permite utilizar diferentes modelos de machine learning para analizar datos y obtener predicciones. Cada modelo está diseñado para diferentes tipos de análisis, desde regresiones simples hasta clasificaciones complejas.

## Preparación de Datos

Antes de comenzar, necesitarás preparar tus datos en un archivo CSV con el siguiente formato:

```plaintext
Entrenamiento;Punto;Euclideano;Manhattan
"A,9,8,8,I;B,2,1,2,III;C,9,8,9,I;D,8,9,9,II;E,1,8,2,I;F,9,8,8,III;G,2,2,3,II";"H,5,7,9";"II";"I,II"
```

### Importante:
- Usa punto y coma (;) como separador entre campos
- Usa comas (,) como separador dentro de cada campo
- Encierra los datos complejos entre comillas dobles (")

## Modelos Disponibles

### 1. Regresión Lineal
- **Uso**: Predecir valores basados en relaciones lineales
- **Datos necesarios**: Pares de valores X,Y para entrenamiento
- **Ejemplo de uso**: Predecir precios basados en características

### 2. Regresión Polinómica
- **Uso**: Predecir valores con relaciones no lineales
- **Datos necesarios**: Valores X,Y y punto a predecir
- **Ejemplo de uso**: Análisis de tendencias complejas

### 3. Árbol de Decisión
- **Uso**: Clasificación basada en reglas
- **Datos necesarios**: Atributos y etiquetas de clase
- **Ejemplo de uso**: Categorización de datos

### 4. Red Neuronal
- **Uso**: Problemas complejos de clasificación/predicción
- **Datos necesarios**: Valores numéricos de entrada
- **Ejemplo de uso**: Reconocimiento de patrones

### 5. K-Means
- **Uso**: Agrupamiento de datos
- **Datos necesarios**: 
  - Número de clusters
  - Datos de entrenamiento
  - Número de iteraciones
- **Ejemplo de uso**: Segmentación de clientes

### 6. K-Nearest Neighbor
- **Uso**: Clasificación basada en proximidad
- **Datos necesarios**:
  - Datos de entrenamiento
  - Punto a clasificar
  - Métricas (Euclideana/Manhattan)
- **Ejemplo de uso**: Clasificación por similitud

## Formato del CSV

Para cada modelo, el archivo CSV debe seguir una estructura específica:

1. **Regresión Lineal y Polinómica**:
```plaintext
x,y
1,2
2,4
3,6
```

2. **K-Nearest Neighbor**:
```plaintext
Entrenamiento;Punto;Euclideano;Manhattan
```

## Uso de los Modelos

1. **Selección del Modelo**:
   - Elige el modelo apropiado según tu necesidad
   - Prepara tus datos en el formato correcto

2. **Carga de Datos**:
   - Asegúrate de que tu CSV esté bien formateado
   - Carga el archivo en el sistema

3. **Ejecución**:
   - Selecciona los parámetros necesarios
   - Ejecuta el modelo
   - Espera los resultados

4. **Interpretación de Resultados**:
   - Los resultados se mostrarán según el modelo:
     - Regresiones: valores predichos
     - Clasificaciones: categorías asignadas
     - Clustering: grupos identificados

## Solución de Problemas

### Problemas Comunes y Soluciones

1. **Error en la Carga del CSV**
   - Verifica el formato del archivo
   - Asegúrate de usar los separadores correctos
   - Comprueba que las comillas estén bien colocadas

2. **Resultados Inesperados**
   - Revisa la calidad de tus datos de entrada
   - Verifica que estés usando el modelo adecuado
   - Comprueba que los datos estén normalizados si es necesario

3. **Error en el Formato de Datos**
   - Asegúrate de que los números usen el formato correcto (punto como decimal)
   - Verifica que no haya espacios extra
   - Comprueba que los datos estén completos

### Recomendaciones

- Haz una copia de seguridad de tus datos antes de procesarlos
- Comienza con conjuntos pequeños de datos para pruebas
- Documenta los parámetros utilizados en cada análisis
- Mantén un registro de los resultados obtenidos

Para más ayuda o soporte técnico, contacta con el equipo de soporte o consulta la documentación técnica detallada.