# -*- coding: utf-8 -*-
# Plantilla de Pre-procesado
# Importar las librerías

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Si no tienen las librerías instaladas, se instalan
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install scikit-learn

#----------------------------------------------------------------------------------------------
# Importar el dataset
dataset = pd.read_csv("Data.csv")
# X -> Matriz de características - Matriz de variables independientes
X = dataset.iloc[:, :-1].values  #Incluimos todas la filas y todas la columnas excepto la última
# y -> variable dependiente - vector a predecir
y = dataset.iloc[:, 3].values

#----------------------------------------------------------------------------------------------
# Tratamiento de los NAs - datos faltantes
# Imputer es una libreria de sklearn poderosa para el manejo de datos faltantes
from sklearn.impute import SimpleImputer
# Se cambiarán los valores nan por la media de la columna (verbose=0)
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose=0)
# Se toman las columnas numéricas 1 y 2, por lo tanto, se pone 3 porque el límite superior no lo toma
imputer = imputer.fit(X[:,1:3]) 
X[:, 1:3] = imputer.transform(X[:,1:3])

# Para versión de python 3.5
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = "nan", strategy = "mean", axis = 0)
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

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
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)
# Como la columna Purchased tiene solo dos categorias (No y Yes), no es necesario
# usar el OneHotEncoder, solo el LabelEncoder para convertirla en 0 y 1, respectivamente
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Para versión de python 3.5
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# De esta forma los convierte en datos ordinales, es decir, datos numéricos con orden
#from sklearn.preprocessing import LabelEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#----------------------------------------------------------------------------------------------
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
# test_size toma el 20% de los datos para testing
# random_state es un número cualquiera para que siempre me genere el mismo resultado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#----------------------------------------------------------------------------------------------
# Escalado de datos
# Cuando dos columnas no se encuentran en el mismo rango (Una muy superior a otra), 
# se debe hacer un escalado de datos
# Escalar -> Normalizar los datos para que los datos estén en el mismo rango
# Estandarización -> Xstand = (x - mean(x))/(standard_desviation(x))
# Normalización -> Xnorm = (x - min(x))/(max(x) - min(x))
# StandardScaler usa la ecuación de Normalización
# Estos usan la ecuación euclidiana (distancia entre dos puntos)
# Inclusive escala las variables dummy
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# Con transform detecta el mismo algoritmo que usó con fit_transform
X_test = sc_X.transform(X_test)

# Este es un algoritmo de clasificación no requiere normalizar la variable y
# Cuando es un algoritmo de predicción se recomienda usar la normalización de la variable a predecir