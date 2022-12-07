import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)

print("###################################################################")
print("1. Carga el fichero **biodeg.data** (es un archivo de texto).")
print("###################################################################")

df = pd.read_csv("S17 - Boosting\\biodeg.data", sep=";", header=0)
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
print("3. Obten resultados utilizando árboles de decisión, bagging y boosting")
print("###################################################################")

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

# boosting
sys_boos_dt = AdaBoostClassifier()
sys_boos_dt.fit(X_train, y_train)
y_pred = sys_boos_dt.predict(X_test)
print("Boosting(árbol)   -    Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# Buscar mejor learning rate para adaboost
val_lr = [0.001, 0.01, 0.1, 1, 10, 20]  # vamos a probar varios valores de learning_rate

# almacenamos todos los resultados en un vector
scores = np.zeros(len(val_lr), dtype=np.float32)

idx_lr = 0
for lr in val_lr:
    print("#### LR:", lr)
    sys_boos_dt.set_params(learning_rate=lr)  # cambiamos el learning rate
    sys_boos_dt.fit(X_train, y_train)  # entrenamos
    y_pred = sys_boos_dt.predict(X_test)  # evaluamos
    scores[idx_lr] = metrics.accuracy_score(y_test, y_pred)
    idx_lr = idx_lr + 1

lrs = [str(p) for p in val_lr]  # creo los ticks del eje x

fig, ax = plt.subplots()
ax.plot(lrs, scores)
ax.set_title("Accuracy vs. learning_rate")
ax.set_xlabel("learning_rate")
ax.set_ylabel("Accuracy")
plt.show()

# La mejor es 1.
# entrenamos de nuevo el AdaBoost
sys_boos_dt = AdaBoostClassifier(learning_rate=1)
sys_boos_dt.fit(X_train, y_train)
y_pred = sys_boos_dt.predict(X_test)
print("Boosting(árbol)   -    Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

print("###################################################################")
print("4. Representa la relevancia de los atributos")
print("###################################################################")

# obtenemos el número de atributos
(num_ejemplos, num_atributos) = X.shape

importances = sys_boos_dt.feature_importances_
# ordenamos los atributos en orden descendente de importancia
indices = np.argsort(importances)[::-1]

# los representamos gráficamente
fig, ax = plt.subplots()
ax.set_title("Relevancia de los atributos AdaBoost")
# [::-1] para que aparezcan en orden decreciente en la gráfica
ax.barh(
    range(num_atributos),
    importances[indices[::-1]],
    tick_label=X.columns[indices[::-1]],
)
plt.show()

print("\n##########################################")
print("### Relevancia con RandomForest")
print("##########################################")

importances = sys_rf.feature_importances_
# ordenamos los atributos en orden descendente de importancia
indices = np.argsort(importances)[::-1]

# los representamos gráficamente
fig, ax = plt.subplots()
ax.set_title("Relevancia de los atributos RandomForest")
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
print("5.3. Obten resultados utilizando árboles de decisión, bagging y boosting")
print("###################################################################")


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

# boosting
sys_boos_dt = AdaBoostClassifier()
sys_boos_dt.fit(X_train, y_train)
y_pred = sys_boos_dt.predict(X_test)
print("Boosting(árbol)   -    Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# Buscar mejor learning rate para adaboost
val_lr = [0.001, 0.01, 0.1, 1, 10, 20]  # vamos a probar varios valores de learning_rate

# almacenamos todos los resultados en un vector
scores = np.zeros(len(val_lr), dtype=np.float32)

idx_lr = 0
for lr in val_lr:
    print("#### LR:", lr)
    sys_boos_dt.set_params(learning_rate=lr)  # cambiamos el learning rate
    sys_boos_dt.fit(X_train, y_train)  # entrenamos
    y_pred = sys_boos_dt.predict(X_test)  # evaluamos
    scores[idx_lr] = metrics.accuracy_score(y_test, y_pred)
    idx_lr = idx_lr + 1

lrs = [str(p) for p in val_lr]  # creo los ticks del eje x

fig, ax = plt.subplots()
ax.plot(lrs, scores)
ax.set_title("Accuracy vs. learning_rate")
ax.set_xlabel("learning_rate")
ax.set_ylabel("Accuracy")
plt.show()

# La mejor es 1.
# entrenamos de nuevo el AdaBoost
sys_boos_dt = AdaBoostClassifier(learning_rate=1)
sys_boos_dt.fit(X_train, y_train)
y_pred = sys_boos_dt.predict(X_test)
print("Boosting(árbol)   -    Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

print("###################################################################")
print("5.4. Representa la relevancia de los atributos")
print("###################################################################")


# obtenemos el número de atributos
(num_ejemplos, num_atributos) = X.shape

importances = sys_boos_dt.feature_importances_
# ordenamos los atributos en orden descendente de importancia
indices = np.argsort(importances)[::-1]

# los representamos gráficamente
fig, ax = plt.subplots()
ax.set_title("Relevancia de los atributos AdaBoost")
# [::-1] para que aparezcan en orden decreciente en la gráfica
ax.barh(
    range(num_atributos),
    importances[indices[::-1]],
    tick_label=X.columns[indices[::-1]],
)
plt.show()

print("\n##########################################")
print("### Relevancia con RandomForest")
print("##########################################")

importances = sys_rf.feature_importances_
# ordenamos los atributos en orden descendente de importancia
indices = np.argsort(importances)[::-1]

# los representamos gráficamente
fig, ax = plt.subplots()
ax.set_title("Relevancia de los atributos RandomForest")
# [::-1] para que aparezcan en orden decreciente en la gráfica
ax.barh(
    range(num_atributos),
    importances[indices[::-1]],
    tick_label=X.columns[indices[::-1]],
)
plt.show()
