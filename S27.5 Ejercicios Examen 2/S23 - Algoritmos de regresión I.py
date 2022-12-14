import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def entrena_y_evalua(X_train, X_test, y_train, y_test, sistema):
    if sistema == "DummyRegressor":
        sys = DummyRegressor(strategy="mean")
    elif sistema == "LinearRegression":
        sys = LinearRegression()
    elif sistema == "Ridge":
        sys = Ridge()
    elif sistema == "PolynomialRegression":
        sys = Pipeline(
            [("pf", PolynomialFeatures(degree=2)), ("linr", LinearRegression())]
        )
    elif sistema == "PolynomialRidge":
        sys = Pipeline(
            [("pf", PolynomialFeatures(degree=2)), ("ridge", Ridge())])
    else:
        print("Sistema no reconocido")
        exit()

    print("\n############################################")
    print("#### %s" % sys)
    print("############################################")

    sys.fit(X_train, y_train)
    y_pred = sys.predict(X_test)

    # error absoluto medio
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print("%s -> MAE = %.3f" % (sys, mae))

    # error cuadrático medio
    mse = metrics.mean_squared_error(y_test, y_pred)
    print("%s -> MSE = %.3f" % (sys, mse))

    # coeficiente de determinación
    r2 = metrics.r2_score(y_test, y_pred)
    print("%s -> R2 = %.3f" % (sys, r2))

    return [mae, mse, r2]


print("###################################################################")
print("1. Carga los datos del fichero **housing.data**.")
print("###################################################################")

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
df = pd.read_csv(
    "S23 - Algoritmos de regresión I\housing.data", names=cabecera, sep="\s+"
)
filas, columnas = df.shape

X = df.iloc[:, 0: (columnas - 1)]
y = df.iloc[:, (columnas - 1)]
print(X)
print(y)


print("###################################################################")
print("2. Separa el conjunto de datos para hacer un hold out 80-20.")
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

print("###################################################################")
print(
    "3. Evalúa los siguientes sistemas (con sus hiperparámetros por defecto) sobre esa partición: sistema media, regresión lineal (con y sin regularización) y regresión polinomial (con y sin regularización)."
)
print("###################################################################")

sistemas = [
    "DummyRegressor",
    "LinearRegression",
    "Ridge",
    "PolynomialRegression",
    "PolynomialRidge",
]

resultados = np.empty((len(sistemas), 3))
i = 0
for sistema in sistemas:
    resultados[i, :] = entrena_y_evalua(
        X_train, X_test, y_train, y_test, sistema)
    i = i + 1

print("###################################################################")
print(
    "4. Muestra, en forma de tabla, los resultados de las tres métricas explicadas en el notebook."
)
print("###################################################################")

df_resultados = pd.DataFrame(
    resultados, index=sistemas, columns=["MAE", "MSE", "R2"])
print(df_resultados)

print("###################################################################")
print(
    "5. Finalmente, calcula el error cuadrático medio que obtiene una regresión polinomial en una validación cruzada de 5 folds."
)
print("###################################################################")

# Pipeline con polynomialfeatures y regresión lineal.
sys_pf_linr = Pipeline(
    [
        ("pf", PolynomialFeatures(degree=2, include_bias=True)),
        ("ridge", LinearRegression()),
    ]
)

# generador de folds partiendo el conjunto en 5 trozos.
folds = KFold(n_splits=5, shuffle=True, random_state=1234)

# validación cruzada para el DummyRegressor.
scores = cross_val_score(
    sys_pf_linr, X, y, cv=folds, scoring="neg_mean_squared_error", verbose=1
)

print(
    "\nRegresión Polinomial neg_MSE (mean +- std): %0.4f +- %0.4f"
    % (scores.mean(), scores.std())
)
print(
    "Regresión Polinomial MSE (mean +- std): %0.4f +- %0.4f"
    % (-scores.mean(), scores.std())
)
