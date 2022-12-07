import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("###################################################################")
print("1. Carga el fichero **Phishing.csv** (es un archivo de texto).")
print("###################################################################")

df = pd.read_csv("S15 - Sesgo y varianza/Phishing.csv", header=0)
filas, columnas = df.shape

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

print("###################################################################")
print(
    "2. Utilizando un 90% de los ejemplos para entrenar comprueba cómo afecta `max_depth` a un árbol de decisión y la `C` a una regresión logística"
)
print("###################################################################")

sys_tree = DecisionTreeClassifier(random_state=1234)

num_profundidades = 25
num_repeticiones = 10

scores = np.zeros((num_repeticiones, num_profundidades), dtype=np.float32)

for rep in range(num_repeticiones):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=rep, stratify=y
    )

    for p in range(1, num_profundidades + 1):
        sys_tree.set_params(max_depth=p)
        sys_tree.fit(X_train, y_train)
        y_pred = sys_tree.predict(X_test)
        scores[rep][p - 1] = metrics.accuracy_score(y_test, y_pred)

depth = [str(p) for p in range(num_profundidades, 0, -1)]
medias = np.mean(scores, axis=0)

plt.plot(depth, medias[::-1])
plt.title("Accuracy vs. max_depth")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.show()


std_lr = Pipeline(
    [("std", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
)
val_C = [0.001, 0.01, 0.1, 1, 10, 100]
num_repeticiones = 10

scores = np.zeros((num_repeticiones, len(val_C)), dtype=np.float32)

for rep in range(num_repeticiones):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=rep, stratify=y
    )

    idx_C = 0
    for C in val_C:
        std_lr.set_params(lr__C=C)
        std_lr.fit(X_train, y_train)
        y_pred = std_lr.predict(X_test)
        scores[rep][idx_C] = metrics.accuracy_score(y_test, y_pred)
        idx_C = idx_C + 1

Cs = [str(p) for p in val_C]
medias = np.mean(scores, axis=0)

plt.plot(Cs, medias)
plt.title("Accuracy vs. C")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.show()


print("###################################################################")
print(
    "3. Utilizando un 10% de los ejemplos para entrenar en 50 repeticiones comprueba la varianza de un árbol de decisión y una regresión logística con sus hiperparámetros por defecto."
)
print("###################################################################")

sys_tree = DecisionTreeClassifier(random_state=1234)
std_lr = Pipeline(
    [("std", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
)

num_repeticiones = 50

scores = np.zeros((num_repeticiones, 2), dtype=np.float32)

for rep in range(num_repeticiones):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=rep, stratify=y
    )

    std_lr.fit(X_train, y_train)
    y_pred = std_lr.predict(X_test)
    scores[rep][0] = metrics.accuracy_score(y_test, y_pred)

    sys_tree.fit(X_train, y_train)
    y_pred = sys_tree.predict(X_test)
    scores[rep][1] = metrics.accuracy_score(y_test, y_pred)

plt.plot(range(num_repeticiones), scores[:, 0], label="reglog")
plt.plot(range(num_repeticiones), scores[:, 1], label="árbol")
plt.title("Varianza en el rendimiento")
plt.xlabel("Repetición")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

print("Reglog - Accuracy: %.4f +- %.4f" % (np.mean(scores[:, 0]), np.std(scores[:, 0])))
print("Árbol - Accuracy: %.4f +- %.4f" % (np.mean(scores[:, 1]), np.std(scores[:, 1])))
