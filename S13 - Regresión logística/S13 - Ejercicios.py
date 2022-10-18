import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import pydotplus


print("############################################################################")
print("1. Carga el fichero **biodeg.data** (es un archivo de texto). ")
print("############################################################################")

print("############################################################################")
print("2. Separar el conjunto en 70% para entrenar y 30% para test (estratificado)")
print("############################################################################")

print("############################################################################")
print("3. Crea 3 sistemas: baseline clase mayoritaria, K vecinos y regresión logística")
print("############################################################################")


print(
    "#################################################################################################################################"
)
print(
    "4. Haz una búsqueda de hiperparámetros (GridSearchCV()) utilizando los ejemplos del conjunto de entrenamiento. Prueba con diferente número de vecinos en el KNN y con diferentes valores de C en la regresión logística."
)
print(
    "#################################################################################################################################"
)


print("############################################################################")
print("5. Comprueba la accuracy de los tres sistemas en el conjunto de test.")
print("############################################################################")

print("############################################################################")
print("6. Dibuja la curva ROC y muestra el AUC.")
print("############################################################################")
