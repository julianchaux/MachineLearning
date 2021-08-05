# -*- coding: utf-8 -*-
# Regresión Polinómica

# Importar las librerías

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----------------------------------------------------------------------------------------------
# Importar el dataset
dataset = pd.read_csv("Position_Salaries.csv")
# X -> Matriz de características - Matriz de variables independientes
X = dataset.iloc[:, 1:2].values  #Incluimos todas la filas y todas la columnas excepto la última
# La variable X debe ser una matriz y NO un vector, por lo tanto, se extrae la columna 1 hasta la 2-1, es decir solo la 1, pero en matriz
# y -> variable dependiente - vector a predecir
y = dataset.iloc[:, 2].values
# y si debe ser un vector

#----------------------------------------------------------------------------------------------
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
# NO SE VA A DIVIDIR PORQUE EL CONJUNTO DE DATOS ES MUY PEQUEÑO

#----------------------------------------------------------------------------------------------
# Ajustar el modelo de Regresión Lineal con todo el conjunto de datos
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lin_regression = LinearRegression()
lin_regression.fit(X, y)
# Hallamos los coeficientes del modelo de regresión
print(f'Los coeficientes de las variables independientes son {lin_regression.coef_}')
print(f'El intercepto de la recta con el eje Y es {lin_regression.intercept_}')

#----------------------------------------------------------------------------------------------
# Ajustar el modelo de Regresión Polinómica con todo el conjunto de datos
from sklearn.preprocessing import PolynomialFeatures
# Regresión Polinómica de grado 2
poly_regression = PolynomialFeatures(degree = 2)
X_poly = poly_regression.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)