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
# Construir el modelo óptimo de Regresión Lineal Múltiple utilizando la Eliminación hacia Atrás solamente con p-valores:
# Técnica de la eliminación hacia atrás para optimizar el modelo
# Con estas técnicas se establece cuáles variables independientes nos dan los mejores resultados
# Puede ser que con menos variables o con las mismas
# El p-valor mide la probabilidad que un coeficiente (b0, b1, b2, bn) en la ecuación de Regresión Lineal Múltiple
# y = b0 + b1x1 + b2x2 + ... + bnxn
import statsmodels.api as sm

def backwardElimination(x, sl):    
    numVars = len(x[0]) 
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)