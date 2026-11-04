# 🛵 Zomato Delivery Operations: Análisis de Causa Raíz, Inferencia Estadística y Modelado Predictivo

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3.0-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Contexto del Problema y Objetivos de Negocio

Zomato enfrenta variaciones significativas en el rendimiento de su red de distribución urbana y en las calificaciones que los usuarios asignan a los repartidores. Este proyecto busca transformar registros operativos crudos en un sistema de analítica diagnóstica y predictiva orientado a la toma de decisiones estratégicas.

### 🎯 Preguntas Problema
1. **Evaluación de Rendimiento:** ¿Qué factores operativos (tráfico, clima, estado del vehículo, distancia y ruteo) determinan la satisfacción del usuario y las bajas calificaciones de los repartidores?
2. **Optimización de Cobertura:** ¿Cómo influyen los cuellos de botella temporales y espaciales en los tiempos de espera para estructurar un plan prescriptivo de mejora operativa?

---

## 🗄️ 2. Ingesta de Datos y Arquitectura del Pipeline

El dataset primario de operaciones de entrega fue extraído e integrado mediante una arquitectura híbrida de SQL y DataFrames en Python:

```python
# Extracción desde SQLite y pasaje a Pandas
conn = sqlite3.connect('zomato_operations.db')
df_raw.to_sql('envios', conn, if_exists='replace', index=False)
df_sql = pd.read_sql("SELECT * FROM envios", conn)
conn.close()

* Email: nicolaschinomartin@gmail.com
