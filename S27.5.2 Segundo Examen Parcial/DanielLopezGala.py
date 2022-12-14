import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel,
                                       SelectKBest, SelectPercentile,
                                       mutual_info_regression)
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def entrena_y_evalua(X_train, X_test, y_train, y_test, sistema):
    if sistema == "DecisionTreeRegressor":
        sys = DecisionTreeRegressor(random_state=1234)
    elif sistema == "LinearRegression":
        sys = LinearRegression()
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


# Leemos el archivo Excel.
df = pd.read_excel("data.xlsx", sheet_name="Absenteeism_at_work")
filas, columnas = df.shape

# Separamos la clase de los datos.
X = df.drop("Class", axis=1)
y = df["Class"]

print(X.head(5))
print(y)


# Aplicamos un Hold-out 70-30 para ver el rendimiento en producción.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234
)

# Probamos los sistemas en el conjunto de datos.
sistemas = [
    "LinearRegression",
    "DecisionTreeRegressor"
]

# Calculamos el rendimiento de los sistemas en el conjunto de test.
resultados = np.empty((len(sistemas), 3))
i = 0
for sistema in sistemas:
    resultados[i, :] = entrena_y_evalua(
        X_train, X_test, y_train, y_test, sistema)
    i = i + 1

df_resultados = pd.DataFrame(
    resultados, index=sistemas, columns=["MAE", "MSE", "R2"])
print(df_resultados)


# Volvemos a crear y entrenar el sistema para obtener los coeficientes.
sys = LinearRegression()
sys.fit(X_train, y_train)
# obtenemos el número de atributos
(num_ejemplos, num_atributos) = X.shape

importances = sys.coef_
# ordenamos los atributos en orden descendente de importancia
indices = np.argsort(importances)[::-1]
# los representamos gráficamente
plt.figure()
plt.title("Coeficientes de los atributos en la regresión lineal.")
# [::-1] para que aparezcan en orden decreciente en la gráfica
plt.barh(range(num_atributos),
         importances[indices[::-1]], tick_label=X.columns[indices[::-1]])
plt.show()

# Como se puede ver en la figura anterior, hay 4 atributos muy importantes en los que el algoritmo basa casi toda su decisión. Estos son el 3, el 5, el 2, y el 8.

# En base a esto, deducimos que podemos utilizar un sistema de selección de atributos. Es decir, aplicar Feature Selection, ya que sólo hay unos pocos atributos que tengan relevancia para predecir la clase.

# Entrenamos Lasso.
sys = Lasso(alpha=1)
sys.fit(X_train, y_train)

seleccionados = sys.feature_names_in_[abs(sys.coef_) > 0.00001]
print("Atributos seleccionados por Lasso (%d):" % len(seleccionados))
print(seleccionados)
importances = sys.coef_
# ordenamos los atributos en orden descendente de importancia
indices = np.argsort(importances)[::-1]
# los representamos gráficamente
plt.figure()
plt.title("Coeficientes de los atributos en Lasso.")
# [::-1] para que aparezcan en orden decreciente en la gráfica
plt.barh(range(num_atributos),
         importances[indices[::-1]], tick_label=X.columns[indices[::-1]])
plt.show()

# se evalúa el rendimiento del Lasso
y_pred = sys.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print("RESULTADOS DE LASSO: ")
print("MAE = %.3f\nMSE = %.3f\nR2 = %.3f" % (mae, mse, r2))
# Como podemos ver, obtiene resultados ligeramente mejores que la regresión lineal.

# Pregunta 1.
# De los sistemas capaces de seleccionar atributos, se escoge Lasso porque es uno de los más populares y mejores resultados obtiene ante estos problemas, como hemos podido ver. Este algoritmo introduce la norma l1 como término de regularización durante el aprendizaje. También podríamos haber utilizado la regresión lineal anterior como selector de atributos usando SelectFromModel, pero Lasso hace una buena tarea de selección de atributos, por lo que no se usa.

# Pregunta 2.
# Si al seleccionar atributos el rendimiento mejora, puede deberse a que el algoritmo se fija en los atributos que más ayudan a predecir la clase sin afectar otros atributos que hagan "ruido". Por ejemplo, un árbol de decisión debería ver mejorado su rendimiento, normalmente, al introducir una selección correcta y favorable de los atributos.
# Si en cambio, el rendimiento disminuye, puede que más atributos de los que estamos usando tras la selección tuviesen cierta relevancia en la predicción de la clase, y nuestro modelo ahora tenga menos información que antes para tratar de averiguar el resultado. Esto puede ocurrir si reducimos el número de atributos para acelerar el proceso de aprendizaje, sin que estos fuesen necesariamente redundantes o innecesarios.
