from sklearn import datasets
import pandas as pd

print("####################################################")
print("###### 1. Cargando algún conjunto desde Sklearn")
print("####################################################")

cjto = datasets.load_diabetes()
X = cjto.data
Y = cjto.target

print(cjto.DESCR)

n_samples, n_features = X.shape
print("Número de ejemplos: ", n_samples)
print("Número de atributos: ", n_features)


print("########################################################")
print("###### 2. Generando un conjunto artificial - regresión")
print("########################################################")

X, Y, coeficientes = datasets.make_regression(
    n_samples=20, n_features=6, n_informative=4, n_targets=1, coef=True
)

print("Coeficientes: \n", coeficientes)
print("Matriz de datos: \n", X)
print("Clase: \n", Y)

print("###########################################################")
print("###### 2. Generando un conjunto artificial - clasificación")
print("###########################################################")

X, Y = datasets.make_classification(
    n_samples=20,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=3,
    weights=[0.5, 0.2, 0.3],
)

print("Matriz de datos: \n", X)
print("Clase: \n", Y)


print("####################################################")
print("###### 3. Cargando desde un fichero de texto.")
print("####################################################")

cabecera = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]

df = pd.read_csv("S01 - Carga de datos/housing.data", sep="\s+", names=cabecera)
filas, columnas = df.shape

X = df.iloc[:, 0 : (columnas - 1)]
print(X)

X_np = X.values

Y = df.iloc[:, (columnas - 1)]
print(Y)

Y_np = Y.values


print("####################################################")
print("###### 3. Cargando desde una hoja de cálculo")
print("####################################################")

df = pd.read_excel("S01 - Carga de datos/Test.xlsx", header=0)
filas, columnas = df.shape
print(df)

X = df.iloc[:, 0 : (columnas - 1)]
print(X)
X_np = X.values

Y = df.iloc[:, (columnas - 1)]
print(Y)
y_np = Y.values
