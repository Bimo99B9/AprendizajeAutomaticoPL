from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
from sklearn import tree

print("############################################################################")
print(
    "1. Carga el mismo conjunto de ejemplos que hemos utilizado en esta práctica añadiendo los ejemplos ruidosos."
)
print("############################################################################")

df = pd.read_excel(
    "S14 - Subajuste, sobreajuste y ruido\ejemplo.xlsx", sheet_name="datos"
)
df_ruido = pd.read_excel(
    "S14 - Subajuste, sobreajuste y ruido\ejemplo.xlsx", sheet_name="ruido"
)
df = pd.concat([df, df_ruido], axis=0, ignore_index=True)
filas, columnas = df.shape

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

print("############################################################################")
print("2. Separar el conjunto en 50% para entrenar y 50% para test (estratificado).")
print("############################################################################")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1234, stratify=y
)


print("############################################################################")
print(
    "3. Utiliza K-vecinos y un árbol de decisión para aprender y evaluar el rendimiento. Visualiza el árbol de decisión."
)
print("############################################################################")

# KNN y su accuracy.
knn_sis = KNeighborsClassifier(n_neighbors=1)
knn_sis.fit(X_train, y_train)
y_pred = knn_sis.predict(X_test)
print("Accuracy KNN: ", metrics.accuracy_score(y_test, y_pred))

# Árbol y su accuracy
sys_dt = DecisionTreeClassifier(random_state=1234)
sys_dt.fit(X_train, y_train)
y_pred = sys_dt.predict(X_test)
print("Accuracy Árbol: ", metrics.accuracy_score(y_test, y_pred))

# Visualizar el árbol
dot_data = tree.export_graphviz(
    decision_tree=sys_dt, feature_names=X.columns, class_names=["0", "1"], filled=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("arbol.png")

print("############################################################################")
print(
    "4. Haz una búsqueda de hiperparámetros (`GridSearchCV()`) utilizando los ejemplos del conjunto de entrenamiento. "
)
print("############################################################################")

n_neighbors = range(1, 11, 1)
max_depth = range(1, 11, 1)
min_impur = [0.0001, 0.001, 0.01, 0.1]

hyperparameters = dict(max_depth=max_depth, min_impurity_decrease=min_impur)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

gs_knn = GridSearchCV(
    knn_sis,
    param_grid={"n_neighbors": n_neighbors},
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

gs_tree

gs_knn_trained = gs_knn.fit(X_train, y_train)
gs_tree_trained = gs_tree.