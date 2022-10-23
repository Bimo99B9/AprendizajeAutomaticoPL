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
from scipy.stats import randint, uniform
import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus

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
    "2. Realiza una validación cruzada de 5 folds con tres árboles: i) árbol por defecto, ii) árbol con `max_depth=3` y iii) árbol con `min_impurity_decrease=0.008`. Compara los resultados entre sí y con los obtenidos por un baseline."
)
print("###################################################################")

# Árbol por defecto
sys_dt = DecisionTreeClassifier(random_state=1234)
# Árbol limitando la profundidad
sys_dt_depth = DecisionTreeClassifier(random_state=1234, max_depth=3)
# Árbol con min_impurity_decrease=0
sys_dt_impurity = DecisionTreeClassifier(random_state=1234, min_impurity_decrease=0.008)
# Se crea el baseline "clase más frecuente".
baseline = DummyClassifier(strategy="most_frequent")

# Realizar la validación cruzada.
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
scores_dt = cross_val_score(sys_dt, X, y, cv=folds, scoring="accuracy")
scores_dt_depth = cross_val_score(sys_dt_depth, X, y, cv=folds, scoring="accuracy")
scores_dt_impurity = cross_val_score(
    sys_dt_impurity, X, y, cv=folds, scoring="accuracy"
)
scores_baseline = cross_val_score(baseline, X, y, cv=folds, scoring="accuracy")

# Resultados
print(
    "\nÁrbol por defecto (mean +- std): %0.4f +- %0.4f"
    % (scores_dt.mean(), scores_dt.std())
)
print(
    "Árbol limitado en profundidad (mean +- std): %0.4f +- %0.4f"
    % (scores_dt_depth.mean(), scores_dt_depth.std())
)
print(
    "Árbol mejora en impureza (mean +- std): %0.4f +- %0.4f"
    % (scores_dt_impurity.mean(), scores_dt_impurity.std())
)
print(
    "Baseline most_frequent (mean +- std): %0.4f +- %0.4f"
    % (scores_baseline.mean(), scores_baseline.std())
)

print("###################################################################")
print(
    "3. Haz una `RandomizedSearchCV` para buscar los mejores hiperparámetros. Que `max_depth`  tome valores entre 1 y 3 y que `min_impurity_decrease` tome valores entre 0 y 0.03."
)
print("###################################################################")

dist_max_depth = randint(1, 5)
dist_min_impurity_decrease = uniform(loc=0, scale=0.03)
hyperparameters = dict(
    max_depth=dist_max_depth, min_impurity_decrease=dist_min_impurity_decrease
)
rs = RandomizedSearchCV(
    sys_dt,
    hyperparameters,
    scoring="accuracy",
    random_state=1234,
    n_iter=100,
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

res_rs = rs.fit(X, y)

# Resultados
print("Hiperparámetros: ", res_rs.cv_results_["params"])
print("Accuracy: ", res_rs.cv_results_["mean_test_score"])
print("Mejor combinación de hiperparámetros: ", res_rs.best_params_)
print("Mejor rendimiento obtenido: %0.4f" % res_rs.best_score_)

print("###################################################################")
print("4. Muestra el árbol que se obtiene con los mejores hiperparámetros")
print("###################################################################")

dot_data = export_graphviz(
    decision_tree=rs.best_estimator_,
    feature_names=X.columns,
    class_names=["0", "1"],
    filled=True,
)

graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png("árbol best estimator.png")
