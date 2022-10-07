print("############################################################################")
print(
    "1. Carga el fichero **heart_failure_clinical_records_dataset.csv** (es un archivo de texto)."
)
print("############################################################################")

from ast import Param
from random import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
    ParameterGrid,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


df = pd.read_csv(
    "S11 - Evaluación del modelo tras ajustar hiperparámetros/heart_failure_clinical_records_dataset.csv",
    header=0,
)

X = df.drop(["DEATH_EVENT"], axis=1)
y = df["DEATH_EVENT"]

print("############################################################################")
print(
    "2. Utiliza los dos métodos vistos en esta sesión para calcular el rendimiento del sistema tras la búsqueda de hiperparámetros."
)
print("############################################################################")

std_sca = StandardScaler()
knn_sis = KNeighborsClassifier()
std_knn = Pipeline([("std", std_sca), ("knn", knn_sis)])

print(
    "### Método 1: GridSearchCV o RandomizedSearchCV dentro de una validación cruzada."
)

weights = ["uniform", "distance"]
p = [1, 2, 3]
n_neighbors = range(2, 11, 2)

hyperparameters = dict(knn__weights=weights, knn__p=p, knn__n_neighbors=n_neighbors)
folds5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

gs = GridSearchCV(
    std_knn, hyperparameters, scoring="accuracy", cv=folds5, verbose=1, n_jobs=-1
)

folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

scores_knn = cross_val_score(gs, X, y, cv=folds10, scoring="accuracy", verbose=1)
print("KNN (mean+-std): %0.4f +- %0.4f" % (scores_knn.mean(), scores_knn.std()))

# Ejecutar búsqueda sin validación cruzada para obtener la mejor combinación.
res_gs = gs.fit(X, y)
print("Mejor combinación de hiperparámetros: ", res_gs.best_params_)
print(f"Mejor accuracy: {res_gs.best_score_}")

print("### Método 2: Entrenamiento - validación - test")

# Separar training y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1234, stratify=y
)

# Separar training en train y val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1234, stratify=y_train
)

weights = ["uniform", "distance"]
p = [1, 2, 3, 4, 5, 6, 7, 8]
n_neighbors = range(2, 12, 1)
hyperparameters = dict(knn__weights=weights, knn__p=p, knn__n_neighbors=n_neighbors)

grid = ParameterGrid(hyperparameters)

best_acc = 0
best_hyperparams = None

for hyperparams in grid:
    std_knn.set_params(**hyperparams)
    std_knn.fit(X_tr, y_tr)
    y_val_pred = std_knn.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    if acc > best_acc:
        best_acc = acc
        best_hyperparams = hyperparams

print(f"Mejores hiperparámetros: {best_hyperparams}")
print(f"Mejor accuracy con ese modelo sobre conjunto de validación: {best_acc}")

# Probar ese modelo sobre conjunto de test
best_model = std_knn.set_params(**best_hyperparams)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)
print(f"Accuracy sobre casos no vistos: {accuracy_score(y_test, y_test_pred)}")
