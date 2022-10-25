import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from scipy.stats import randint

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
    "2. Realiza una búsqueda aleatoria utilizando valores para el número de vecinos entre 1 y 50, para la `p` de Minkowski valores entre 1 y 10 y ponderando o no las distancias. **OJO**, ten en cuenta que los atributos tienen escalas diferentes, así que deberás crear un pipeline."
)
print("###################################################################")

# Crear el Pipeline de KNN estandarizado.
std_sca = StandardScaler()
knn_sis = KNeighborsClassifier()
std_knn = Pipeline([("std", std_sca), ("knn", knn_sis)])

# Realizar búsqueda aleatoria.
dist_n_neighbors = randint(1, 50)
dist_p = randint(1, 10)
weights = ["uniform", "distance"]

hyperparameters = dict(
    knn__weights=weights, knn__p=dist_p, knn__n_neighbors=dist_n_neighbors
)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

rs = RandomizedSearchCV(
    std_knn,
    hyperparameters,
    scoring="accuracy",
    random_state=1234,
    n_iter=100,
    cv=folds,
    verbose=3,
    n_jobs=-1,
)

res_rs = rs.fit(X, y)

print("Hiperparámetros: ", res_rs.cv_results_["params"])
print("Accuracy: ", res_rs.cv_results_["mean_test_score"])
print("Desviación: ", res_rs.cv_results_["std_test_score"])
print("Mejor combinación de hiperparámetros: ", res_rs.best_params_)
print("Mejor rendimiento obtenido: %0.4f" % (res_rs.best_score_))

print("###################################################################")
print(
    "3. Una vez acotado el espacio de búsqueda, realiza una búsqueda más exhaustiva utilizando una `GridSearchCV()`."
)
print("###################################################################")

# Obtener los valores obtenidos anteriormente
best_nn = res_rs.best_params_["knn__n_neighbors"]
best_p = res_rs.best_params_["knn__p"]
print("Mejor número de vecinos: ", best_nn)
print("Mejor p: ", best_p)

# Crear listas con valores en ese entorno.
n_neighbors = range(max(1, best_nn - 4), best_nn + 5, 2)
p = range(max(1, best_p - 2), best_p + 3)
print("Vecinos a explorar     :", list(n_neighbors))
print("Valores de p a explorar:", list(p))

hyperparameters = dict(knn__weights=weights, knn__p=p, knn__n_neighbors=n_neighbors)

gs = GridSearchCV(
    std_knn, hyperparameters, scoring="accuracy", cv=folds, verbose=3, n_jobs=-1
)

res_gs = gs.fit(X, y)

print("Hiperparámetros: ", res_gs.cv_results_["params"])
print("Accuracy: ", res_gs.cv_results_["mean_test_score"])
print("Desviación: ", res_gs.cv_results_["std_test_score"])
print("Mejor combinación de hiperparámetros: ", res_gs.best_params_)
print("Mejor rendimiento obtenido: %0.4f" % (res_gs.best_score_))
