# -*- coding: utf-8 -*-
# Regresión Lineal Múltiple

# Importar las librerías

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----------------------------------------------------------------------------------------------
# Importar el dataset
dataset = pd.read_csv("50_Startups.csv")
# X -> Matriz de características - Matriz de variables independientes
X = dataset.iloc[:, :-1].values  #Incluimos todas la filas y todas la columnas excepto la última
# y -> variable dependiente - vector a predecir
y = dataset.iloc[:, 4].values

#----------------------------------------------------------------------------------------------
# Codificar datos categóricos
# Convertir datos de texto (categorías) en números para poder procesarlos
# Variable Dummy (One Hot Encode) -> Convierte una categoría que no tiene orden
# a un vector del número de categorías que hayan, por ejemplo, en este caso son
# tres categorías, por lo tanto, el vector dummy tiene tres columnas y pone un 1
# en la columna de la categoría -> Un solo uno por fila para la categoria
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Para versión de python 3.5
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

# Debemos evitar la trampa de las variables ficticias (Dummy)
# Recordar que cuando trabajamos con variables Dummy se deben quitar una de estas variables
# para evitar la multicolinealidad, es decir, si son tres (3) datos categóricos, habrían 
# tres (3) columnas Dummy, por lo tanto se toman dos (2) para el modelo
# Eliminamos la primera columna X[0] que es la primera columna Dummy
X = X[:, 1:]

#----------------------------------------------------------------------------------------------
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
# test_size toma el 20% de los datos para testing
# random_state es un número cualquiera para que siempre me genere el mismo resultado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#----------------------------------------------------------------------------------------------
# Ajustar el modelo de Regresión Lineal Múltiple con el conjunto de datos de entrenamiento
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
regression = LinearRegression()
regression.fit(X_train, y_train)
# Hallamos los coeficientes del modelo de regresión
print(f'Los coeficientes de las variables independientes son {regression.coef_}')
print(f'El intercepto de la recta con el eje Y es {regression.intercept_}')

# Cálculo del error cuadrático medio (MSE) de los *y_train* respecto a los *y* del modelo de predicción
mse_train = mean_squared_error(y_train, regression.predict(X_train))
print(f'El MSE de los datos de entrenamiento es {mse_train}')

# Cálculo del coeficiente de determinación (r^2) de los *y_train* respecto a los *y* del modelo de predicción
rcuadrado_train = r2_score(y_train, regression.predict(X_train))
print(f'El coeficiente de determinación (r^2) de los datos de entrenamiento es {rcuadrado_train}')

#----------------------------------------------------------------------------------------------
# Predecir los resultados (Regresión Lineal Múltiple) con el conjunto de datos de test
y_pred = regression.predict(X_test)


# Cálculo del error cuadrático medio (MSE) de los *y_test* respecto a los *y* del modelo de predicción
mse_test = mean_squared_error(y_test, y_pred)
print(f'El MSE de los datos de prueba es {mse_test}')

# Cálculo del coeficiente de determinación (r^2) de los *y_test* respecto a los *y* del modelo de predicción
rcuadrado_test = r2_score(y_test, y_pred)
print(f'El coeficiente de determinación (r^2) de los datos de prueba es {rcuadrado_test}')

#----------------------------------------------------------------------------------------------
# Construir el modelo óptimo de Regresión Lineal Múltiple utilizando la Eliminación hacia Atrás
# Técnica de la eliminación hacia atrás para optimizar el modelo
# Con estas técnicas se establece cuáles variables independientes nos dan los mejores resultados
# Puede ser que con menos variables o con las mismas
# El p-valor mide la probabilidad que un coeficiente (b0, b1, b2, bn) en la ecuación de Regresión Lineal Múltiple
# y = b0 + b1x1 + b2x2 + ... + bnxn
import statsmodels.api as sm
# Teniendo en cuenta que el p-valor mide la probabilidad que un coeficiente sea cero (0)
# y en la matriz de variables independientes no está incluido el valor de b0, se simulará
# añadiendo una columna de unos (1) para incluirlo en el análisis
# Esta nueva columna se añade al inicio de X
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# ELIMINACIÓN HACIA ATRÁS
# 1er paso: Creamos el nivel de significación SL
SL = 0.05

# X_opt es la nueva matriz de variables independientes estadísticamente significativas
# que optimiza la predicción del modelo
# 2do paso: Se inicia  con todas la variables independientes
X_opt = X[:, [0, 1, 2, 3, 4, 5]].tolist()

# Crear el OLS (Ordinary List Square - Mínimos cuadrados ordinarios) con todas las variables predictoras
# Con este paso se vuelve a calcular la Regresión Lineal Múltiple, pero con información
# de cual es la próxima variable indpendiente a sacar del modelo optimizado
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# 3er paso: Examinar el p-valor.  Si es mayor a SL se saca de la matriz X_opt
regression_OLS.summary()
# En los resultados observamos que el p-valor para cada variable independiente es
# x0(b0) = 0.000, x1(Dummy1) = 0.953, x2(Dummy2) = 0.990, x3(R&D) = 0.000, x4(Admin) = 0.608, 
# x5(Marketing) = 0.123
# Por lo tanto eliminamos la variable independiente con el p-valor mas grande ya que es mayor que SL

# 4o paso: Eliminamos la columna x2 y ajustamos el modelo nuevamente
X_opt = X[:, [0, 1, 3, 4, 5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
# En los resultados observamos que el p-valor para cada variable independiente es
# x0(b0) = 0.000, x1(Dummy1) = 0.940, x3(R&D) = 0.000, x4(Admin) = 0.604, 
# x5(Marketing) = 0.118
# Por lo tanto eliminamos la variable independiente con el p-valor mas grande ya que es mayor que SL

# 4o paso: Eliminamos la columna x1 y ajustamos el modelo nuevamente
X_opt = X[:, [0, 3, 4, 5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
# En los resultados observamos que el p-valor para cada variable independiente es
# x0(b0) = 0.000, x3(R&D) = 0.000, x4(Admin) = 0.602, x5(Marketing) = 0.105
# Por lo tanto eliminamos la variable independiente con el p-valor mas grande ya que es mayor que SL

# 4o paso: Eliminamos la columna x4 y ajustamos el modelo nuevamente
X_opt = X[:, [0, 3, 5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
# En los resultados observamos que el p-valor para cada variable independiente es
# x0(b0) = 0.000, x3(R&D) = 0.000, x5(Marketing) = 0.060
# Por lo tanto eliminamos la variable independiente con el p-valor mas grande ya que es mayor que SL

# 4o paso: Eliminamos la columna x5 y ajustamos el modelo nuevamente
X_opt = X[:, [0, 3]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
# En los resultados observamos que el p-valor para cada variable independiente es
# x0(b0) = 0.000, x3(R&D) = 0.000
# Ya encontramos la matriz X óptima usando el criterio del p-valor

# CONCLUSIÓN: Para este caso, la variable R&D es la variable más estadísticamente significativa