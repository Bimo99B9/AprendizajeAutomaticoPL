from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

print("###################################################################")
print("1. Carga el fichero **biodeg.data** (es un archivo de texto).")
print("###################################################################")

df = pd.read_csv("S16 - Bagging\\biodeg.data", sep=";", header=0)
filas, columnas = df.shape
print(df.head(5))

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

print("###################################################################")
print("2. Separa el conjunto en un 70% para entrenar y un 30% para test.")
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y
)

print("###################################################################")
print(
    "3. Obten resultados utilizando Regresión Logística, árboles de decisión y utilizando bagging"
)
print("###################################################################")

# regresión logística
std_lr = Pipeline([("std", StandardScaler()), ("lr", LogisticRegression())])
std_lr.fit(X_train, y_train)
y_pred = std_lr.predict(X_test)
print("Reglog     -      Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# bagging con regresión logística
sys_bag_rl = BaggingClassifier(base_estimator=std_lr, random_state=1234, n_jobs=-1)
sys_bag_rl.fit(X_train, y_train)
y_pred = sys_bag_rl.predict(X_test)
print("Bagging(RegLog) - Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# árbol de decisión
sys_dt = DecisionTreeClassifier(random_state=1234)
sys_dt.fit(X_train, y_train)
y_pred = sys_dt.predict(X_test)
print("Árbol de decisión - Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# random forest
sys_rf = RandomForestClassifier(random_state=1234, n_jobs=-1)
sys_rf.fit(X_train, y_train)
y_pred = sys_rf.predict(X_test)
print("RandomForest   -    Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

print("###################################################################")
print("4. Representa la relevancia de los atributos")
print("###################################################################")

# obtenemos el número de atributos
(num_ejemplos, num_atributos) = X.shape

importances = sys_rf.feature_importances_
# ordenamos los atributos en orden descendente de importancia
indices = np.argsort(importances)[::-1]

# los representamos gráficamente
fig, ax = plt.subplots()
fig.set_size_inches((16, 12))
ax.set_title("Relevancia de los atributos (RF)")
# [::-1] para que aparezcan en orden decreciente en la gráfica
ax.barh(
    range(num_atributos),
    importances[indices[::-1]],
    tick_label=X.columns[indices[::-1]],
)
plt.show()

print("###################################################################")
print("5. Repite lo mismo con el conjunto **ionosphere.data**")
print("###################################################################")

cabecera = ["atr" + str(x) for x in range(1, 35)]
cabecera.append("clase")
df = pd.read_csv("S16 - Bagging\ionosphere.data", names=cabecera)
filas, columnas = df.shape
print(df.head(5))

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

print("###################################################################")
print("5.2. Separa el conjunto en un 70% para entrenar y un 30% para test.")
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y
)

print("###################################################################")
print(
    "5.3. Obten resultados utilizando Regresión Logística, árboles de decisión y utilizando bagging"
)
print("###################################################################")

# regresión logística
std_lr = Pipeline([("std", StandardScaler()), ("lr", LogisticRegression())])
std_lr.fit(X_train, y_train)
y_pred = std_lr.predict(X_test)
print("Reglog     -      Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# bagging con regresión logística
sys_bag_rl = BaggingClassifier(base_estimator=std_lr, random_state=1234, n_jobs=-1)
sys_bag_rl.fit(X_train, y_train)
y_pred = sys_bag_rl.predict(X_test)
print("Bagging(RegLog) - Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# árbol de decisión
sys_dt = DecisionTreeClassifier(random_state=1234)
sys_dt.fit(X_train, y_train)
y_pred = sys_dt.predict(X_test)
print("Árbol de decisión - Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# random forest
sys_rf = RandomForestClassifier(random_state=1234, n_jobs=-1)
sys_rf.fit(X_train, y_train)
y_pred = sys_rf.predict(X_test)
print("RandomForest   -    Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

print("###################################################################")
print("5.4. Representa la relevancia de los atributos")
print("###################################################################")

# obtenemos el número de atributos
(num_ejemplos, num_atributos) = X.shape

importances = sys_rf.feature_importances_
# ordenamos los atributos en orden descendente de importancia
indices = np.argsort(importances)[::-1]

# los representamos gráficamente
fig, ax = plt.subplots()
fig.set_size_inches((16, 12))
ax.set_title("Relevancia de los atributos (RF)")
# [::-1] para que aparezcan en orden decreciente en la gráfica
ax.barh(
    range(num_atributos),
    importances[indices[::-1]],
    tick_label=X.columns[indices[::-1]],
)
plt.show()
