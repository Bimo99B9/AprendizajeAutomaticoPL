import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("###################################################################")
print("1. Carga el fichero **biodeg.data** (es un archivo de texto).")
print("###################################################################")

df = pd.read_csv("S18 - Máquinas de Vectores Soporte\\biodeg.data", sep=";", header=0)
filas, columnas = df.shape
print(df.head(5))

class_enc = LabelEncoder()

df["experimental class"] = class_enc.fit_transform(df["experimental class"])

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

print(df.head(5))

print("###################################################################")
print(
    "2. Busca el mejor valor de C y evalúa el rendimiento final del modelo (utiliza una `GridSearchCV` dentro de una validación cruzada como se vio en la sesión 11). Observa que valores altos de C pueden hacer que los experimentos sean interminables"
)
print("###################################################################")

sys_svc = SVC(kernel="linear")

Cs = [0.001, 0.01, 0.1, 1, 10, 100]

folds5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

gs = GridSearchCV(
    sys_svc, param_grid={"C": Cs}, scoring="accuracy", cv=folds5, verbose=1, n_jobs=-1
)

folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

scores_scv = cross_val_score(
    gs, X, y, cv=folds10, scoring="accuracy", verbose=1, n_jobs=-1
)

print("\nSVC (mean+-std): %0.4f +- %0.4f" % (scores_scv.mean(), scores_scv.std()))

res_gs = gs.fit(X, y)
print("Mejor combinación de hiperparámetros: ", res_gs.best_params_)


print("###################################################################")
print("3. Estandariza los atributos y repite el paso 2")
print("###################################################################")

std_svc = Pipeline([("std", StandardScaler()), ("svc", SVC(kernel="linear"))])


gs = GridSearchCV(
    std_svc,
    param_grid={"svc__C": Cs},
    scoring="accuracy",
    cv=folds5,
    verbose=1,
    n_jobs=-1,
)

folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

scores_scv = cross_val_score(
    gs, X, y, cv=folds10, scoring="accuracy", verbose=1, n_jobs=-1
)

print("\nSVC (mean+-std): %0.4f +- %0.4f" % (scores_scv.mean(), scores_scv.std()))

res_gs = gs.fit(X, y)
print("Mejor combinación de hiperparámetros: ", res_gs.best_params_)
