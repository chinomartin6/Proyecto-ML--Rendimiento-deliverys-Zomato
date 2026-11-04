# 🚴 Zomato Delivery Operations: Optimización de Tiempos y Calificaciones mediante Machine Learning e Inferencia Estadística

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Modeling-orange.svg)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Inferential%20Stats-green.svg)](https://scipy.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Data%20Ingestion-blueviolet.svg)](https://www.sqlite.org/)

---

## 📋 Tabla de Contenidos
1. [Contexto del Problema y Pregunta de Negocio](#-contexto-del-problema-y-pregunta-de-negocio)
2. [Arquitectura del Dataset e Ingesta de Datos](#-arquitectura-del-dataset-e-ingesta-de-datos)
3. [Análisis Exploratorio de Datos (EDA) y Limpieza Espacial](#-análisis-exploratorio-de-datos-eda-y-limpieza-espacial)
4. [Ingeniería de Características (Feature Engineering)](#-ingeniería-de-características-feature-engineering)
5. [Auditoría de Calidad y Prevención de Data Leakage](#-auditoría-de-calidad-y-prevención-de-data-leakage)
6. [Modelado Predictivo de Machine Learning (Fase 8)](#-modelado-predictivo-de-machine-learning-fase-8)
7. [Diagnóstico de Causa Raíz e Inferencia Estadística (Fase 9)](#-diagnóstico-de-causa-raíz-e-inferencia-estadística-fase-9)
8. [Conclusiones y Recomendaciones Prescriptivas de Negocio (Fase 10)](#-conclusiones-y-recomendaciones-prescriptivas-de-negocio-fase-10)
9. [Hoja de Ruta Técnica (Evolución v2.0)](#-hoja-de-ruta-técnica-evolución-v20)
10. [Instrucciones de Reproducibilidad](#-instrucciones-de-reproducibilidad)

---

## 🎯 Contexto del Problema y Pregunta de Negocio

En las plataformas de entrega a domicilio (*Quick-Commerce* / *Food Delivery*), la satisfacción del cliente impacta de forma directa sobre la retención de usuarios y el *Lifetime Value* (LTV). Zomato identificó una fluctuación en las evaluaciones otorgadas a los repartidores, lo que motivó el desarrollo de este pipeline analítico end-to-end.

* **Pregunta Problema Central:** ¿Cuáles son las causas raíz operativas y de entorno que provocan bajas calificaciones en los repartidores de Zomato?
* **Objetivos de Negocio:**
  1. Cuantificar el peso predictivo de las variables logísticas (tiempo, distancia, tráfico, clima, flota) sobre la valoración recibida por el repartidor (`delivery_person_ratings`).
  2. Identificar y validar estadísticamente las cuellos de botella que incrementan el tiempo de entrega (`time_taken_(min)`).
  3. Formular un plan de acción prescriptivo con impacto directo en la operación y en la arquitectura algorítmica de la app.

---

## 🗄️ Arquitectura del Dataset e Ingesta de Datos

El dataset procesado corresponde a las operaciones logísticas de Zomato, descargado desde Kaggle Hub (`saurabhbadole/zomato-delivery-operations-analytics-dataset`).

* **Pipeline de Ingesta:** Para simular un entorno de arquitectura empresarial de datos, el dataset crudo se importó a una base de datos **SQLite** (`zomato_operations.db`), volcando la tabla `envios` hacia DataFrames de Pandas mediante consultas SQL optimizadas (`SELECT * FROM envios`).
* **Dimensiones Iniciales:** 45,584 registros y 20 columnas operativas.

---

## 🧼 Análisis Exploratorio de Datos (EDA) y Limpieza Espacial

El EDA se dividió en cuatro ejes metodológicos principales:

### 1. Auditoría de Calidad y Estandarización de Nulos
Se identificaron nulos enmascarados como texto (`"NaN"`, `"N/A"`, `"nan"`, `"-1"`). Se estandarizaron a valores matemáticos nulos de NumPy (`np.nan`). Las variables estratégicas fueron imputadas bajo criterios rigurosos:
* **Variables Numéricas:** Imputación por Mediana (`delivery_person_ratings`, `delivery_person_age`).
* **Variables Categóricas:** Imputación por Moda (`weather_conditions`, `road_traffic_density`).
* **Variables Temporales:** Descarte directo de filas con `time_orderd` nulo para no generar sesgo en la imputación de la hora de emisión.

### 2. Tratamiento de Errores Geográficos (*Null Island*)
Mediante análisis de dispersión espacial (*Scatter plots*), se detectó una anomalia severa de recolección de datos: coordenadas situadas exactamente en $(0.0, 0.0)$ (latitud 0, longitud 0). Se depuraron estructuralmente estas filas para evitar distorsiones kilométricas en el cálculo del trayecto.

```python
# Depuración estricta de Null Island (0.0, 0.0)
df = df[
    (df['restaurant_latitude'] != 0.0) &
    (df['restaurant_longitude'] != 0.0) &
    (df['delivery_location_latitude'] != 0.0) &
    (df['delivery_location_longitude'] != 0.0)
].copy()
```

### 3. Winsorización y Normalización
* **Outliers:** Se aplicó **Winsorización al 1% y 99%** (`scipy.stats.mstats.winsorize`) sobre `time_taken_(min)` y `delivery_person_ratings`, acotando valores atípicos sin eliminar volumen muestral útil.
* **Escalamiento:** Estandarización Z-score (`StandardScaler`) para crear métricas relativas respecto a la media del grupo.

### 4. Resumen Estadístico Post-EDA
* **Filas finales:** 42,075 registros (reducción de 3,509 filas defectuosas por GPS o nulos irreversibles).
* **Tiempo Promedio de Entrega:** $26.28 \text{ min}$ (Desviación estándar de $35.4\%$).
* **Calificación Promedio:** $4.63 \text{ estrellas}$ (Alta concentración entre $4.5$ y $5.0$).

---

## 🛠️ Ingeniería de Características (Feature Engineering)

Se construyeron nuevas métricas operativas para traducir datos crudos en variables explicativas con alto valor estratégico:

1. **Distancia Haversine (`distancia_km`):** Cálculo de la distancia esférica real entre las coordenadas del restaurante y el cliente:
   $$d = 2r \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
2. **Velocidad Promedio (`velocidad_kmh`):** Métrica objetiva que combina `distancia_km` y `time_taken_(min)`.
3. **Rango de Rapidez (`rango_rapidez`):** Discretización operativa: *Lento* ($<15 \text{ km/h}$), *Estándar* ($15\text{--}30 \text{ km/h}$), *Rápido* ($>30 \text{ km/h}$).
4. **Banderas Temporales:** Extracción del día (`dia_semana`), indicador binario de fin de semana (`fin_de_semana`) y hora entera del pedido (`hora_pedido`).
5. **Entorno Operativo:**
   * **`clima_adverso`:** Variable binaria reestructurada mediante expresiones regulares (`stormy|sandstorms|fog|rainy|windy`).
   * **`trafico_nivel`:** Codificación ordinal de la densidad del tráfico ($1 = \text{Ligero}$, $2 = \text{Transitable}$, $3 = \text{Congestionado}$, $4 = \text{Embotellamiento}$).

---

## 🚨 Auditoría de Calidad y Prevención de Data Leakage

Durante las pruebas iniciales de modelado, se detectó un valor inusual de $R^2 = 1.0$ y $\text{MAE} = 0.0$. 

### Diagnóstico y Resolución
Una auditoría del *Feature Importance* reveló que la columna `delivery_person_ratings_zscore` concentraba el $100\%$ del peso predictivo por ser una transformación matemática directa de la variable objetivo ($y$). 

* **Acción Correctiva (Fase 7):** Se excluyeron de la matriz $X$ las columnas `delivery_person_ratings` y `delivery_person_ratings_zscore`.
* **Train-Test Split:** División limpia 80% entrenamiento (32,275 filas) y 20% prueba (8,069 filas) utilizando la semilla aleatoria `SEED = 42`.

---

## 📊 Modelado Predictivo de Machine Learning (Fase 8)

Se entrenaron y evaluaron cuatro algoritmos de regresión supervisada para predecir la calificación del repartidor (`delivery_person_ratings`):

| Modelo | MAE | RMSE | $R^2$ Score | Desempeño |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **0.1824** | **0.2541** | **0.4163** | **Ganador Absoluto** |
| **Gradient Boosting Regressor** | 0.1882 | 0.2598 | 0.3895 | Competitivo |
| **Árbol de Decisión** | 0.1891 | 0.2614 | 0.3820 | Susceptible a overfitting |
| **Regresión Lineal (Baseline)** | 0.2281 | 0.3082 | 0.1410 | Fracaso por no linealidad |

### Insights Técnicos del Modelado
1. **El cliente valora el tiempo total de espera:** La variable `time_taken_(min)` domina el *Feature Importance* con más del **$44\%$ del peso predictivo total**. La distancia ($8.3\%$) y la velocidad ($4.8\%$) tienen un rol secundario.
2. **Operación por Umbrales de Tolerancia:** La Regresión Lineal fracasó ($R^2 = 14.1\%$) porque asumía que cada minuto extra restaba nota de forma constante. Random Forest ganó ($R^2 = 41.63\%$) al capturar que pequeñas variaciones en entregas rápidas ($15\text{--}20 \text{ min}$) no alteran la nota, pero al superar los $30\text{--}35 \text{ minutos}$, la calificación se desploma de forma no lineal.
3. **Capacidad Explicativa:** El modelo explica el **$41.63\%$ de la varianza total** de las calificaciones con un error promedio de apenas $0.18$ estrellas ($\text{MAE} = 0.1824$).

---

## 🔬 Diagnóstico de Causa Raíz e Inferencia Estadística (Fase 9)

Dado que la calificación ($y$) depende directamente del tiempo de entrega ($X_{\text{principal}}$), la Fase 9 evaluó las causas raíz estructurales que generan demoras, aplicando pruebas inferenciales formales.

$$\text{Factores Operativos } (\text{Tráfico, Vehículo, Clima, Calendario}) \longrightarrow \text{Tiempo de Entrega } (X_{\text{principal}}) \longrightarrow \text{Calificación } (y)$$

```
=====================================================================================
PRUEBAS DE INFERENCIA ESTADÍSTICA (EVALUACIÓN DE SIGNIFICANCIA OPERATIVA)
=====================================================================================
• ANOVA Tráfico vs. Tiempo:            Estadístico F =  3033.52 | p-valor = 0.00e+00
• T-Test Fin de Semana vs. Tiempo:     Estadístico t =     5.02 | p-valor = 5.20e-07
• T-Test Clima Favorable vs. Adverso:  Estadístico t =   -12.91 | p-valor = 4.75e-38
• ANOVA Estado Vehículo vs. Tiempo:    Estadístico F =  1733.31 | p-valor = 0.00e+00
```

### Resumen de Hallazgos Inferenciales

1. **Densidad de Tráfico ($F = 3033.52, p = 0.00$):**
   * Correlación con tiempo: $r = 0.4160$.
   * Pasar de Tráfico Ligero ($21.29 \text{ min}$) a Embotellamiento / *Jam* ($31.07 \text{ min}$) genera un **sobrecosto operativo de $+9.78 \text{ minutos}$ (+45.94% de demora)**.
2. **Entregas Múltiples ($r = 0.3861$):**
   * Asignar más de un pedido por viaje es el segundo factor que más infla el tiempo, evidenciando que el ahorro por ruteo agrupado deteriora el nivel de servicio.
3. **Condición del Vehículo ($F = 1733.31, p = 0.00$):**
   * Vehículos en Mal Estado ($0$): $29.98 \text{ min}$ promedio.
   * Vehículos Regular ($1$) o Bueno ($2$): $\sim 24.4 \text{ min}$ promedio.
   * **Salto crítico:** La penalización ocurre únicamente al caer a la categoría $0$, sumando **$+5.6 \text{ minutos}$ de retraso por viaje**. Entre estado $1$ y $2$ no existen diferencias operativas significativas.
4. **Clima Adverso ($t = -12.91, p = 4.75 \times 10^{-38}$):**
   * Clima Favorable: $25.43 \text{ min}$ vs. Clima Adverso: $26.70 \text{ min}$.
   * Provoca un sobrecosto de **$+1.27 \text{ minutos}$ (+4.99% de demora)**.
5. **Estructura Semanal ($t = 5.02, p = 5.20 \times 10^{-7}$):**
   * Los días laborales ($26.42 \text{ min}$) resultaron ser **$0.53 \text{ minutos}$ más lentos** que el fin de semana ($25.89 \text{ min}$), impulsados por la congestión comercial de los miércoles y viernes.

---

## 💡 Conclusiones y Recomendaciones Prescriptivas de Negocio (Fase 10)

Para transformar los hallazgos analíticos en decisiones de negocio con impacto operativo, se proponen cuatro intervenciones prescriptivas para Zomato:

### 1. Promesa de Tiempo Dinámica (Algoritmo ETA Inteligente)
* **Acción:** Ajustar el cálculo del tiempo estimado de llegada (ETA) que visualiza el usuario en la app.
* **Justificación:** Cuando el sistema detecte tráfico denso (`High` o `Jam`) o clima adverso, debe ampliar automáticamente el tiempo prometido. Si el usuario recibe un ETA realista de $35 \text{ min}$ en lugar de $20 \text{ min}$, la expectativa se ajusta y la nota no cae al cruzar el umbral de tolerancia.

### 2. Estándar Mínimo de Flota (Filtro de Mantenimiento Vehicular)
* **Acción:** Implementar revisiones técnicas obligatorias para vehículos calificados en condición $0$ (Malo).
* **Justificación:** La inferencia estadística probó que un vehículo degradado retrasa el pedido en $+5.6 \text{ minutos}$, mientras que llevarlo a condición regular ($1$) restituye toda la eficiencia operativa.

### 3. Tope a las Entregas Múltiples en Horas Pico
* **Acción:** Restringir la asignación de pedidos simultáneos a un solo repartidor cuando la densidad de tráfico sea `Congestionado` o `Embotellamiento`.
* **Justificación:** La correlación de $r = 0.3861$ demuestra que el ahorro económico del ruteo múltiple destruye valor al penalizar la velocidad y bajar la calificación del cliente.

### 4. Compensación Meteorológica Automatizada
* **Acción:** Incrementar automáticamente la ventana de entrega en $+1.27 \text{ minutos}$ durante alertas de tormenta, niebla o lluvia.

---

## 🚀 Hoja de Ruta Técnica (Evolución v2.0)

Para la siguiente iteración del proyecto analítico, se recomienda:
* **Aislamiento del Tiempo de Cocina:** Limpiar y calcular el delta entre `time_orderd` y `time_order_picked` para separar los retrasos del restaurante de los retrasos en ruta.
* **Captura de Varianza No Explicada ($58.4\%$):** Incorporar en la app una micro-encuesta post-entrega que mida el estado de la comida (temperatura/empaque) y la amabilidad del repartidor, factores invisibles en la tabla de datos actual.

---

## 🛠️ Instrucciones de Reproducibilidad

1. **Clonar el repositorio:**
   ```bash
   git clone https:[//github.com/tu-usuario/zomato-delivery-analytics.git](https://github.com/chinomartin6/Proyecto-ML--Rendimiento-deliverys-Zomato)
   cd Rendimiento-deliverys-Zomato
   ```

2. **Instalar dependencias:**
   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib seaborn kagglehub
   ```

3. **Ejecutar el pipeline:**
   * Abrir y ejecutar secuencialmente el cuaderno Jupyter `proyecto_zomato.ipynb` en Google Colab o Jupyter Lab.
   * El script descargará automáticamente el dataset vía `kagglehub`, creará la base SQLite local y ejecutará desde la Fase 1 hasta la Fase 9 sin interrupciones.

---

*Desarrollado como proyecto integral de Ciencia de Datos y Analítica Diagnóstica de Negocio.*

* Email: nicolaschinomartin@gmail.com
