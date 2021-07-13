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
from sklearn.metrics import mean_squared_error, r2_score
regression = LinearRegression()
regression.fit(X_train, y_train)
# Hallamos los coeficientes del modelo de regresión
print(f'El coeficiente de la variable independiente es {regression.coef_}')
print(f'El intercepto de la recta con el eje Y es {regression.intercept_}')

# Cálculo del error cuadrático medio (MSE) de los *y_train* respecto a los *y* del modelo de predicción
mse = mean_squared_error(y_train, regression.predict(X_train))
print(f'El MSE de los datos de entrenamiento es {mse}')

# Cálculo del coeficiente de determinación (r^2) de los *y_train* respecto a los *y* del modelo de predicción
rcuadrado = r2_score(y_train, regression.predict(X_train))
print(f'El coeficiente de determinación (r^2) de los datos de entrenamiento es {rcuadrado}')

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

# Cálculo del error cuadrático medio (MSE) de los *y_test* respecto a los *y* del modelo de predicción
mse = mean_squared_error(y_test, y_pred)
print(f'El MSE de los datos de prueba es {mse}')

# Cálculo del coeficiente de determinación (r^2) de los *y_test* respecto a los *y* del modelo de predicción
rcuadrado = r2_score(y_test, y_pred)
print(f'El coeficiente de determinación (r^2) de los datos de prueba es {rcuadrado}')