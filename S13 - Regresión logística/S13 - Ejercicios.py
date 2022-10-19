import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import pydotplus


print("############################################################################")
print("1. Carga el fichero **biodeg.data** (es un archivo de texto). ")
print("############################################################################")

df = pd.read_csv("S13 - Regresión logística/biodeg.data", sep=";")
filas, columnas = df.shape

print(df.head(5))

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]


print("############################################################################")
print("2. Separar el conjunto en 70% para entrenar y 30% para test (estratificado)")
print("############################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y
)

print("############################################################################")
print("3. Crea 3 sistemas: baseline clase mayoritaria, K vecinos y regresión logística")
print("############################################################################")

cl_my_sis = DummyClassifier(strategy="most_frequent")

std_knn = Pipeline([("std", StandardScaler()), ("knn", KNeighborsClassifier())])

std_lr = Pipeline(
    [("std", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
)

print(
    "#################################################################################################################################"
)
print(
    "4. Haz una búsqueda de hiperparámetros (GridSearchCV()) utilizando los ejemplos del conjunto de entrenamiento. Prueba con diferente número de vecinos en el KNN y con diferentes valores de C en la regresión logística."
)
print(
    "#################################################################################################################################"
)

n_neighbors = range(1, 12, 1)
c = [0.0001, 0.001, 0.001, 0.1, 1, 10, 100]

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

gs_lr = GridSearchCV(
    std_lr,
    param_grid={"lr__C": c},
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

gs_knn = GridSearchCV(
    std_knn,
    param_grid={"knn__n_neighbors": n_neighbors},
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)


gs_lr_trained = gs_lr.fit(X_train, y_train)
gs_knn_trained = gs_knn.fit(X_train, y_train)
cl_my_sis_trained = cl_my_sis.fit(X_train, y_train)


print("LR - Hyperparams: ", gs_lr_trained.best_params_)
print("LR - Mejor accuracy en GridSearch: %.4f" % gs_lr_trained.best_score_)
print("KNN - Hyperparams: ", gs_knn_trained.best_params_)
print("KNN - Mejor accuracy en GridSearch: %.4f" % gs_knn_trained.best_score_)


print("############################################################################")
print("5. Comprueba la accuracy de los tres sistemas en el conjunto de test.")
print("############################################################################")

y_pred = gs_lr_trained.predict(X_test)
print("LR - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))
y_pred = gs_knn_trained.predict(X_test)
print("KNN - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))
y_pred = cl_my_sis_trained.predict(X_test)
print("Baseline - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))

print("############################################################################")
print("6. Dibuja la curva ROC y muestra el AUC.")
print("############################################################################")
