# -*- coding: utf-8 -*-
# Regresión Lineal Simple

# Importar las librerías

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----------------------------------------------------------------------------------------------
# Importar el dataset
dataset = pd.read_csv("Salary_Data.csv")
# X -> Matriz de características - Matriz de variables independientes
X = dataset.iloc[:, :-1].values  #Incluimos todas la filas y todas la columnas excepto la última
# y -> variable dependiente - vector a predecir
y = dataset.iloc[:, 1].values

#----------------------------------------------------------------------------------------------
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
# test_size toma 1 de cada 3 de los datos para testing
# random_state es un número cualquiera para que siempre me genere el mismo resultado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#----------------------------------------------------------------------------------------------
# Crear el modelo de Regresión Lineal Simple con el conjunto de datos de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#----------------------------------------------------------------------------------------------
# Predecir los datos (Regresión Lineal Simple) con el conjunto de datos de test
y_pred = regression.predict(X_test)

#----------------------------------------------------------------------------------------------
# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de datos de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $US)")
plt.show()

#----------------------------------------------------------------------------------------------
# Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de datos de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $US)")
plt.show()