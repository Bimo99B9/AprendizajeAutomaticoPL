import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    chi2,
    mutual_info_classif,
    f_classif,
)


def filtro(X_train, X_test, y_train, y_test, tipo_sel, relevancias):
    if tipo_sel == "SelectKBest":
        selector = SelectKBest(relevancias, k=10)
    elif tipo_sel == "SelectPercentile":
        selector = SelectPercentile(relevancias, percentile=25)
    else:
        print("Selector incorrecto")
        exit()

    print("\n###########################################")
    print("###" + ts + "(" + rel.__name__ + ")")
    print("###########################################")

    selector.fit(X_train, y_train)

    print("Atributos seleccionados (%d):" % (selector.get_feature_names_out().shape[0]))
    print(selector.get_feature_names_out())

    # Transformar conjuntos de train y test.
    X_train_rel = selector.transform(X_train)
    X_test_rel = selector.transform(X_test)

    # Entrenamos y evaluamos un árbol de decisión con los atributos seleccionados.
    sys_tree.fit(X_train_rel, y_train)
    y_pred = sys_tree.predict(X_test_rel)
    acc_tree = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy árbol   = %.4f" % acc_tree)

    # Entrenamos y evaluamos un knn con los atributos seleccionados.
    sys_knn.fit(X_train_rel, y_train)
    y_pred = sys_knn.predict(X_test_rel)
    acc_knn = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy KNN(k=5) = %.4f" % acc_knn)

    return [acc_tree, acc_knn, selector.get_feature_names_out().shape[0]]


print("###################################################################")
print(
    "1. Carga el conjunto de datos  **optdigits** que viene ya separado en entrenamiento (.tra) y test (.tes). Se trata de un conjunto de clasificación en el que hay 10 clases (se pretende identificar dígitos mediante sus imágenes, consulta el .names)"
)
print("###################################################################")

cabecera = ["b" + str(x) for x in range(1, 65)]
cabecera.append("clase")
df = pd.read_csv(
    "S25 - Reducción de dimensionalidad Feature selection\optdigits.tra", names=cabecera
)
filas, columnas = df.shape

X_train = df.iloc[:, 0 : (columnas - 1)]
y_train = df.iloc[:, (columnas - 1)]

df = pd.read_csv(
    "S25 - Reducción de dimensionalidad Feature selection\optdigits.tes", names=cabecera
)
filas, columnas = df.shape

X_test = df.iloc[:, 0 : (columnas - 1)]
y_test = df.iloc[:, (columnas - 1)]


print("###################################################################")
print(
    "2. Aplica algún filtro de los vistos y compara con los resultados que se obtendrían utilizando todos los atributos con un árbol y un k-vecinos."
)
print("###################################################################")

cor = X_train.corr()
cor_abs = cor.abs()
# print(cor_abs)

# Rendimiento original
print("Rendimiento original: ")

sys_tree = DecisionTreeClassifier(random_state=1234)
sys_tree.fit(X_train, y_train)
y_pred = sys_tree.predict(X_test)
acc_tree = metrics.accuracy_score(y_test, y_pred)
print("Accuracy árbol   = %.4f" % acc_tree)

sys_knn = KNeighborsClassifier()
sys_knn.fit(X_train, y_train)
y_pred = sys_knn.predict(X_test)
acc_knn = metrics.accuracy_score(y_test, y_pred)
print("Accuracy KNN(k=5)   = %.4f" % acc_tree)


# Experimentos y sistemas a probar

tipo_sel = ["SelectKBest", "SelectPercentile"]
relevancias = [mutual_info_classif, chi2, f_classif]

# Matriz para almacenar los resultados
resultados = np.empty((len(tipo_sel) * len(relevancias) + 1, 3))
# Incorporar resultados con todos los atributos.
resultados[0, :] = [acc_tree, acc_knn, X_train.shape[1]]
sistemas = ["Con todos los atributos"]

i = 1
for ts in tipo_sel:
    for rel in relevancias:
        resultados[i, :] = filtro(X_train, X_test, y_train, y_test, ts, rel)
        sistemas.append(ts + "(" + rel.__name__ + ")")
        i = i + 1

df_resultados = pd.DataFrame(
    resultados,
    index=sistemas,
    columns=["Árbol de decisión", "KNN (k=5)", "num atributos"],
)
print(df_resultados)
