from ast import Param
import pandas as pd
from sklearn import impute, metrics, preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    LeaveOneOut,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    ParameterGrid,
)
from scipy.stats import randint
import seaborn as sns
import matplotlib.pyplot as plt


print("###################################################################")
print(
    "1. Carga el fichero **heart_failure_clinical_records_dataset.csv** (es un archivo de texto)."
)
print("###################################################################")

df = pd.read_csv(
    "S10 - Selección de hiperparámetros/heart_failure_clinical_records_dataset.csv",
    header=0,
)
filas, columnas = df.shape

X = df.iloc[:, 0 : (columnas - 1)]
print(X)
y = df.iloc[:, (columnas - 1)]
print(y)

print("###################################################################")
print(
    "2. Utiliza los dos métodos vistos en esta sesión para calcular el rendimiento del sistema tras la búsqueda de hiperparámetros."
)
print("###################################################################")

# Crear Pipeline, (Estandarizador + KNN)
std_sca = StandardScaler()
knn_sis = KNeighborsClassifier()
std_knn = Pipeline([("std", std_sca), ("knn", knn_sis)])

print("Método 1: GridSearchCV o RandomizedSearchCV dentro de una validación cruzada.")
# Se buscan hiperparámetros con GridSearchCV dentro de cada iteración de la validación cruzada.

weights = ["uniform", "distance"]
p = [1, 2, 3]
n_neighbors = range(2, 11, 1)
hyperparameters = dict(knn__weights=weights, knn__p=p, knn__n_neighbors=n_neighbors)

# Se crea la GridSearch para cada iteración de la validación cruzada
folds5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
gs = GridSearchCV(
    std_knn, hyperparameters, scoring="accuracy", cv=folds5, verbose=1, n_jobs=-1
)

# Se introduce en la validación cruzada gs como algoritmo.
folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
scores_knn = cross_val_score(gs, X, y, cv=folds10, scoring="accuracy", verbose=1)
print("\nKNN (mean +- std): %0.4f +- %0.4f" % (scores_knn.mean(), scores_knn.std()))

# Si queremos obtener los hiperparámetros, hay que ejecutar la búsqueda sin validación cruzada.
res_gs = gs.fit(X, y)
print("Mejor combinación de hiperparámetros: ", res_gs.best_params_)


print("\nMétodo 2: Entrenamiento - validación - test.")

# Separar los datos en training set y test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1234, stratify=y
)

# Separar los datos de entrenamiento en train y val para la búsqueda de hiperparámetros
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1234, stratify=y_train
)

# Crear un grid con los parámetros.
weights = ["uniform", "distance"]
p = [1, 2, 3]
n_neighbors = range(2, 11, 1)
hyperparameters = dict(knn__weights=weights, knn__p=p, knn__n_neighbors=n_neighbors)
grid = ParameterGrid(hyperparameters)

# Inicializar las variables para almacenar las mejores.
best_acc = 0
best_hyperparams = None

for hyperparams in grid:
    std_knn.set_params(**hyperparams)
    print(std_knn)
    std_knn.fit(X_tr, y_tr)
    y_val_pred = std_knn.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_val_pred)
    if acc > best_acc:
        best_acc = acc
        best_hyperparams = hyperparams

print("\nMejor combinación de hiperparámetrs: ", best_hyperparams)
print("Mejor rendimiento obtenido sobre el conjunto de validación : %0.4f" % best_acc)


# Asignar los mejores hiperparámetros
best_model = std_knn.set_params(**best_hyperparams)
print(best_model)

# Reentrenar el vecino más próximo
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)
print(
    "\nAccuracy sobre casos no vistos: %0.4f"
    % metrics.accuracy_score(y_test, y_test_pred)
)
