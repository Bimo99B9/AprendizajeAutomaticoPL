import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

print("###################################################################")
print("1. Carga los datos del fichero **housing.data**.")
print("###################################################################")

cabecera = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv(
    "S23 - Algoritmos de regresión I\housing.data", names=cabecera, sep='\s+')
filas, columnas = df.shape

X = df.iloc[:, 0: (columnas - 1)]
y = df.iloc[:, (columnas - 1)]
print(X)
print(y)


print("###################################################################")
print("2. Separa el conjunto de datos para hacer un hold out 80-20.")
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234, stratify=y
)

print("###################################################################")
print("3. Evalúa los siguientes sistemas (con sus hiperparámetros por defecto) sobre esa partición: sistema media, regresión lineal (con y sin regularización) y regresión polinomial (con y sin regularización).")
print("###################################################################")

sistemas = ["DummyRegressor", "LinearRegression",
            "Ridge", "PolynomialRegression", "PolynomialRidge"]

resultados = np.empty((len(sistemas), 3))
i = 0
for sistema in sistemas:
    resultados[i, :] = entrena_y_evalua(
        X_train, X_test, y_train, y_test, sistema)


print("###################################################################")
print("4. Muestra, en forma de tabla, los resultados de las tres métricas explicadas en el notebook.")
print("###################################################################")


print("###################################################################")
print("5. Finalmente, calcula el error cuadrático medio que obtiene una regresión polinomial en una validación cruzada de 5 folds.")
print("###################################################################")
