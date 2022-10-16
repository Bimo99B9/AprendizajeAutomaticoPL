print("############################################################################")
print(
    "1. Carga el fichero **heart_failure_clinical_records_dataset.csv** (es un archivo de texto)."
)
print("############################################################################")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from scipy.stats import randint, uniform


df = pd.read_csv("S12 - Árboles de decisión/heart_failure_clinical_records_dataset.csv")

X = df.drop(["DEATH_EVENT"], axis=1)
y = df["DEATH_EVENT"]

print("############################################################################")
print(
    "2. Realiza una validación cruzada de 5 folds con tres árboles: i) árbol por defecto, ii) árbol con `max_depth=3` y iii) árbol con `min_impurity_decrease=0.008`. Compara los resultados entre sí y con los obtenidos por un baseline."
)
print("############################################################################")

# Crear DecisionTreeClassifier
sys_dt = DecisionTreeClassifier(random_state=1234)
# Crear DecisionTreeClassifier limitando la profundidad
sys_dt_depth = DecisionTreeClassifier(random_state=1234, max_depth=3)
# Crear DecisionTreeClassifier indicando la minima mejora de impureza
sys_dt_impurity = DecisionTreeClassifier(random_state=1234, min_impurity_decrease=0.008)

# Crear baseline "clase más frecuente".
baseline = DummyClassifier(strategy="most_frequent")

## Validaciones cruzadas con 5 folds.

# Se crea un generador de folds estratificados.
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
# Validación cruzada con arbol default.
scores_dt = cross_val_score(sys_dt, X, y, cv=folds, scoring="accuracy")
# Validación cruzada arbol limitando profundidad.
scores_dt_depth = cross_val_score(sys_dt_depth, X, y, cv=folds, scoring="accuracy")
# Validación cruzada arbol minima impureza
scores_dt_impurity = cross_val_score(
    sys_dt_impurity, X, y, cv=folds, scoring="accuracy"
)
# Validación cruzada clase más frecuente.
scores_baseline = cross_val_score(baseline, X, y, cv=folds, scoring="accuracy")

print("Árbol default: %0.4f +- %0.4f" % (scores_dt.mean(), scores_dt.std()))
print(
    "Árbol limitado en profundidad: %0.4f +- %0.4f"
    % (scores_dt_depth.mean(), scores_dt_depth.std())
)
print(
    "Árbol mejora en impureza: %0.4f +- %0.4f"
    % (scores_dt_impurity.mean(), scores_dt_impurity.std())
)
print(
    "BAseline clase más frecuente: %0.4f +- %0.4f"
    % (scores_baseline.mean(), scores_baseline.std())
)


print("############################################################################")
print(
    "3. Haz una `RandomizedSearchCV` para buscar los mejores hiperparámetros. Que `max_depth`  tome valores entre 1 y 3 y que `min_impurity_decrease` tome valores entre 0 y 0.03."
)
print("############################################################################")

# Distribución uniforme entre 1 y 4.
dist_max_depth = randint(1, 5)
# Distribución uniforme entre 0 y 0.03
dist_min_impurity_decrease = uniform(loc=0, scale=0.03)

# Crear diccionario de hiperparámetros
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

print("Mejor combinación de hiperparámetros: ", res_rs.best_params_)
print("Mejor rendimiento obtenido: %0.4f" % res_rs.best_score_)

print("############################################################################")
print("4. Muestra el árbol que se obtiene con los mejores hiperparámetros")
print("############################################################################")

dot_data = export_graphviz(
    decision_tree=rs.best_estimator_,
    feature_names=X.columns,
    class_names=["0", "1"],
    filled=True,
)

graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png("S12 - Árboles de decisión/arbol best estimator.png")
