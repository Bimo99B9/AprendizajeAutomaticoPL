from random import random
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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus

print("###################################################################")
print("1. Carga el fichero **biodeg.data** (es un archivo de texto). ")
print("###################################################################")

df = pd.read_csv("S13 - Regresión logística/biodeg.data", sep=";", header=0)
rows, columns = df.shape
print(df.head(5))

class_enc = LabelEncoder()
df["experimental class"] = class_enc.fit_transform(df["experimental class"])

X = df.iloc[:, 0 : (columns - 1)]
print(X)
y = df.iloc[:, (columns - 1)]
print(y)


print("###################################################################")
print("2. Separar el conjunto en 70% para entrenar y 30% para test (estratificado)")
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y
)

print("###################################################################")
print("3. Crea 3 sistemas: baseline clase mayoritaria, K vecinos y regresión logística")
print("###################################################################")

baseline = DummyClassifier(strategy="most_frequent")
std_knn = Pipeline([("std", StandardScaler()), ("knn", KNeighborsClassifier())])
std_lr = Pipeline(
    [("std", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
)

print("###################################################################")
print(
    "4. Haz una búsqueda de hiperparámetros (`GridSearchCV()`) utilizando los ejemplos del conjunto de entrenamiento. Prueba con diferente número de vecinos en el `KNN` y con diferentes valores de `C` en la regresión logística."
)
print("###################################################################")

valores_n_neighbors = range(1, 11, 1)
valores_de_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Crear el generador de folds estratificados partiendo el conjunto en 5 trozos
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

# Crear las grid search
gs_knn = GridSearchCV(
    std_knn,
    param_grid={"knn__n_neighbors": valores_n_neighbors},
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)
gs_lr = GridSearchCV(
    std_lr,
    param_grid={"lr__C": valores_de_C},
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

# Ejecutar las búsquedas
gs_knn_trained = gs_knn.fit(X_train, y_train)
gs_lr_trained = gs_lr.fit(X_train, y_train)

# Entrenar el baseline
baseline.fit(X_train, y_train)

# Resultados
print("KNN - Mejor combinación de hiperparámetros: ", gs_knn_trained.best_params_)
print("KNN - Mejor rendimiento obtenido: %0.4f" % gs_knn_trained.best_score_)
print("LR - Mejor combinación de hiperparámetros: ", gs_lr_trained.best_params_)
print("LR - Mejor rendimiento obtenido: %0.4f" % gs_lr_trained.best_score_)

print("###################################################################")
print("5. Comprueba la accuracy de los tres sistemas en el conjunto de test.")
print("###################################################################")

# Predicciones en conjunto de test
y_pred_knn = gs_knn_trained.predict(X_test)
y_pred_lr = gs_lr_trained.predict(X_test)
y_pred_bl = baseline.predict(X_test)

print("\nKNN - Accuracy en test: %0.4f" % metrics.accuracy_score(y_test, y_pred_knn))
print("LR - Accuracy en test: %0.4f" % metrics.accuracy_score(y_test, y_pred_lr))
print("Baseline - Accuracy en test: %0.4f" % metrics.accuracy_score(y_test, y_pred_bl))

print("###################################################################")
print("6. Dibuja la curva ROC y muestra el AUC.")
print("###################################################################")

metrics.RocCurveDisplay.from_estimator(gs_knn_trained, X_test, y_test, name="KNN")
plt.show()
metrics.RocCurveDisplay.from_estimator(
    gs_lr_trained, X_test, y_test, name="Logistic Regression"
)
plt.show()
