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
print("1. Carga el fichero **ionosphere.data** (es un archivo de texto).")
print("###################################################################")

cabecera = ["atr" + str(x) for x in range(1, 35)]
cabecera.append("clase")
df = pd.read_csv("S19 - Kernel trick\ionosphere.data", names=cabecera)
filas, columnas = df.shape

class_enc = LabelEncoder()

df["clase"] = class_enc.fit_transform(df["clase"])
print("Clases: ", class_enc.classes_)

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]


print("###################################################################")
print(
    "2. Prueba los kernels polinómico y RBF (prueba diferentes valores para los hiperparámetros `C`, `degree` y `gamma`)."
)
print("###################################################################")

# GridSearch + CrossVal: Kernel Polinómico.
sys_svc = SVC(kernel="poly")

valores_Cs = [0.01, 0.1, 1, 10, 100]
valores_degrees = [1, 2, 3]

hyperparameters = dict(C=valores_Cs, degree=valores_degrees)

folds5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

gs = GridSearchCV(
    sys_svc,
    param_grid=hyperparameters,
    scoring="accuracy",
    cv=folds5,
    verbose=1,
    n_jobs=-1,
)

folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

scores_svc = cross_val_score(
    gs, X, y, cv=folds10, scoring="accuracy", verbose=1, n_jobs=-1
)

print("\nSVC (mean+-std): %0.4f +- %0.4f" % (scores_svc.mean(), scores_svc.std()))

# Búsqueda sin validación para obtener la mejor combinación.
res_gs = gs.fit(X, y)
print("Mejor combinación de hiperparámetros: ", res_gs.best_params_)

# GridSearch + CrossVal: Kernel RBF
sys_svc = SVC(kernel="rbf")

valores_Cs = [0.01, 0.1, 1, 10, 100, 1000]
valores_gammas = [0.01, 0.1, 1, 10, 100]

hyperparameters = dict(C=valores_Cs, gamma=valores_gammas)

folds5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

gs = GridSearchCV(
    sys_svc,
    param_grid=hyperparameters,
    scoring="accuracy",
    cv=folds5,
    verbose=1,
    n_jobs=-1,
)

folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

scores_svc = cross_val_score(
    gs, X, y, cv=folds10, scoring="accuracy", verbose=1, n_jobs=-1
)

print("\nSVC (mean+-std): %0.4f +- %0.4f" % (scores_svc.mean(), scores_svc.std()))

# Búsqueda sin validación para obtener la mejor combinación.
res_gs = gs.fit(X, y)
print("Mejor combinación de hiperparámetros: ", res_gs.best_params_)

print("###################################################################")
print(
    "3. Busca los mejores valores para los hiperparámetros y evalúa el rendimiento final del modelo (utiliza una `GridSearchCV` dentro de una validación cruzada como se vio en la sesión 11). "
)
print("###################################################################")
