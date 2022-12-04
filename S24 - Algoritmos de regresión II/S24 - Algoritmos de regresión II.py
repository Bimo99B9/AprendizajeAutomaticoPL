import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


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
        sys = Pipeline([("pf", PolynomialFeatures(degree=2)), ("ridge", Ridge())])
    elif sistema == "KNeighborsRegressor":
        sys = KNeighborsRegressor()
    elif sistema == "SVM-poly":
        sys = SVR(kernel="poly")
    elif sistema == "SVM-rbf":
        sys = SVR(kernel="rbf")
    elif sistema == "DecisionTreeRegressor":
        sys = DecisionTreeRegressor(random_state=1234)
    elif sistema == "RandomForestRegressor":
        sys = RandomForestRegressor(random_state=1234)
    elif sistema == "AdaBoostRegressor":
        sys = AdaBoostRegressor(random_state=1234)
    elif sistema == "XGBRegressor":
        sys = XGBRegressor()
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
print("1. Carga los datos del fichero **airfoil_self_noise.data**.")
print("###################################################################")

cabecera = [
    "Frequency",
    "Angle of attack",
    "Chord length",
    "Free-stream velocity",
    "Suction",
    "Decibels",
]
df = pd.read_csv(
    "S24 - Algoritmos de regresión II\\airfoil_self_noise.data",
    names=cabecera,
    sep="\t",
)
filas, columnas = df.shape

X = df.iloc[:, 0 : (columnas - 1)]
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
print("3. Evalúa el rendimiento de varios algoritmos de los comentados en esta sesión.")
print("###################################################################")

sistemas = [
    "DummyRegressor",
    "LinearRegression",
    "Ridge",
    "PolynomialRegression",
    "PolynomialRidge",
    "KNeighborsRegressor",
    "SVM-poly",
    "SVM-rbf",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "AdaBoostRegressor",
    "XGBRegressor",
]

resultados = np.empty((len(sistemas), 3))
i = 0
for sistema in sistemas:
    resultados[i, :] = entrena_y_evalua(X_train, X_test, y_train, y_test, sistema)
    i = i + 1

print("###################################################################")
print(
    "4. Muestra, en forma de tabla, los resultados de las métricas explicadas en el notebook."
)
print("###################################################################")

df_resultados = pd.DataFrame(resultados, index=sistemas, columns=["MAE", "MSE", "R2"])
print(df_resultados)
