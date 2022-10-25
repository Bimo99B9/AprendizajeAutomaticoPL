import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

print("###################################################################")
print(
    "1. Carga el mismo conjunto de ejemplos que hemos utilizado en esta práctica añadiendo los ejemplos ruidosos."
)
print("###################################################################")

df = pd.read_excel(
    "S14 - Subajuste, sobreajuste y ruido/ejemplo.xlsx", sheet_name="datos"
)
df_ruido = pd.read_excel(
    "S14 - Subajuste, sobreajuste y ruido/ejemplo.xlsx", sheet_name="ruido"
)
df = pd.concat([df, df_ruido], axis=0, ignore_index=True)
filas, columnas = df.shape

print(df.head(5))

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

print("###################################################################")
print("2. Separar el conjunto en 50% para entrenar y 50% para test (estratificado).")
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1234, stratify=y
)

print("###################################################################")
print(
    "3. Utiliza K-vecinos y un árbol de decisión para aprender y evaluar el rendimiento. Visualiza el árbol de decisión."
)
print("###################################################################")

# KNN
sys_knn = KNeighborsClassifier(n_neighbors=1)
sys_knn.fit(X_train, y_train)
y_pred = sys_knn.predict(X_train)
print("KNN - Accuracy en train: %.4f" % metrics.accuracy_score(y_train, y_pred))
y_pred = sys_knn.predict(X_test)
print("KNN - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))

# STD KNN (No se pide)
std_knn = Pipeline(
    [("std", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=1))]
)
std_knn.fit(X_train, y_train)
y_pred = std_knn.predict(X_train)
print("STDKNN - Accuracy en train: %.4f" % metrics.accuracy_score(y_train, y_pred))
y_pred = std_knn.predict(X_test)
print("STDKNN - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))

# Árbol de decisión
sys_tree = DecisionTreeClassifier(random_state=1234)
sys_tree.fit(X_train, y_train)
y_pred = sys_tree.predict(X_train)
print("Árbol - Accuracy en train: %.4f" % metrics.accuracy_score(y_train, y_pred))
y_pred = sys_tree.predict(X_test)
print("Árbol - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))

# Visualizar árbol de decisión
dot_data = export_graphviz(
    decision_tree=sys_tree, feature_names=X.columns, class_names=["0", "1"], filled=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("árbol sin regularización.png")

print("###################################################################")
print(
    "4. Haz una búsqueda de hiperparámetros (GridSearchCV()) utilizando los ejemplos del conjunto de entrenamiento."
)
print("###################################################################")

# Definir los valores de hiperparámetros a probar.
valores_n_neighbors = range(1, 11, 1)
valores_max_depth = range(1, 11, 1)
valores_min_impurity_decrease = [0.0001, 0.001, 0.01, 0.1]

hyperparameters = dict(
    max_depth=valores_max_depth, min_impurity_decrease=valores_min_impurity_decrease
)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

# Crear las grid search
gs_knn = GridSearchCV(
    sys_knn,
    param_grid={"n_neighbors": valores_n_neighbors},
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

gs_std_knn = GridSearchCV(
    std_knn,
    param_grid={"knn__n_neighbors": valores_n_neighbors},
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

gs_tree = GridSearchCV(
    sys_tree,
    param_grid=hyperparameters,
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

# Ejecutar las búsquedas de hiperparámetros
gs_knn_trained = gs_knn.fit(X_train, y_train)
gs_std_knn_trained = gs_std_knn.fit(X_train, y_train)
gs_tree_trained = gs_tree.fit(X_train, y_train)

# Resultados
print("\nKNN - Mejor combinación de hiperparámetros: ", gs_knn_trained.best_params_)
print("KNN - Mejor rendimiento obtenido: %0.4f" % gs_knn_trained.best_score_)
print(
    "\nSTDKNN - Mejor combinación de hiperparámetros: ", gs_std_knn_trained.best_params_
)
print("STDKNN - Mejor rendimiento obtenido: %0.4f" % gs_std_knn_trained.best_score_)
print(
    "\nÁrbol - Mejor combinación de hiperparámetros: ",
    gs_tree_trained.best_params_,
)
print("Árbol - Mejor rendimiento obtenido: %0.4f" % gs_tree_trained.best_score_)

print("###################################################################")
print("5. Evalúa el rendimiento de los sistemas y visualiza el nuevo árbol generado.")
print("###################################################################")

# Predicciones del KNN
y_pred = gs_knn_trained.predict(X_train)
print("\nKNN - Accuracy en train: %.4f" % metrics.accuracy_score(y_train, y_pred))
y_pred = gs_knn_trained.predict(X_test)
print("KNN - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))
# Predicciones del STDKNN
y_pred = gs_std_knn_trained.predict(X_train)
print("STDKNN - Accuracy en train: %.4f" % metrics.accuracy_score(y_train, y_pred))
y_pred = gs_std_knn_trained.predict(X_test)
print("STDKNN - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))
# Predicciones del árbol
y_pred = gs_tree_trained.predict(X_train)
print("Árbol - Accuracy en train: %.4f" % metrics.accuracy_score(y_train, y_pred))
y_pred = gs_tree_trained.predict(X_test)
print("Árbol - Accuracy en test: %.4f" % metrics.accuracy_score(y_test, y_pred))


# Visualizar el árbol
dot_data = export_graphviz(
    decision_tree=gs_tree_trained.best_estimator_,
    feature_names=X.columns,
    class_names=["0", "1"],
    filled=True,
)
graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("arbol con regularización.png")
