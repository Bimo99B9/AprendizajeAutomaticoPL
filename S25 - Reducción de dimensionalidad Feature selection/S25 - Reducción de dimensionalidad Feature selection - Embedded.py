import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

X_train = df.iloc[:, 0: (columnas - 1)]
y_train = df.iloc[:, (columnas - 1)]

df = pd.read_csv(
    "S25 - Reducción de dimensionalidad Feature selection\optdigits.tes", names=cabecera
)
filas, columnas = df.shape

X_test = df.iloc[:, 0: (columnas - 1)]
y_test = df.iloc[:, (columnas - 1)]


print("###################################################################")
print("4. Prueba la selección que realiza el RandomForest.")
print("###################################################################")

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

# Crear RandomForestClassifier
sys_rf = RandomForestClassifier(random_state=1234)
# Entrenar y calcular relevancia de los atributos.
sys_rf.fit(X_train, y_train)

# Fijando un umbral mayor que la que tendrían si todos fuesen igual de relevantes (1/64).
seleccionados = sys_rf.feature_names_in_[
    sys_rf.feature_importances_ > (1 / 64)]
print("Atributos seleccionados (%d): " % len(seleccionados))
print(seleccionados)

y_pred = sys_rf.predict(X_test)
acc_sys = metrics.accuracy_score(y_test, y_pred)
print("Accuracy RandomForest = %.3f" % (acc_sys))

print("RandomForest como selector de atributos")

selector = SelectFromModel(sys_rf, threshold=(1 / 64))
selector.fit(X_train, y_train)

print("Atributos seleccionados (%d): " %
      (selector.get_feature_names_out().shape[0]))
X_train_rel = selector.transform(X_train)
X_test_rel = selector.transform(X_test)

# Entrenar un árbol de decisión con los atributos seleccionados.
sys_tree.fit(X_train_rel, y_train)
y_pred = sys_tree.predict(X_test_rel)
acc_tree = metrics.accuracy_score(y_test, y_pred)
print("Accuracy KNN(k=5) = %.4f" % acc_knn)
