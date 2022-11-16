import pandas as pd
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

print("###################################################################")
print("1. Carga el fichero **heart_failure_clinical_records_dataset.csv** (es un archivo de texto). ")
print("###################################################################")

df = pd.read_csv(
    "S20 - Naïve Bayes\heart_failure_clinical_records_dataset.csv", header=0)

filas, columnas = df.shape

X = df.iloc[:, 0:(columnas - 1)]
y = df.iloc[:, (columnas-1)]

print("###################################################################")
print("2. Separar el conjunto en 70% para entrenar y 30% para test (estratificado)")
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y)

print("###################################################################")
print("3. Obtén la accuracy del `GaussianNB` con hiperparámetros por defecto.")
print("###################################################################")

sys_nb = GaussianNB()
sys_nb.fit(X_train, y_train)
y_pred = sys_nb.predict(X_test)
print("Accuracy en el conjunto de test: %0.4f" %
      metrics.accuracy_score(y_test, y_pred))


print("###################################################################")
print("4. Haz una búsqueda de `var_smoothing` con `GridSearchCV()` utilizando los ejemplos del conjunto de entrenamiento y evalúa el mejor modelo con el conjunto de test.")
print("###################################################################")

val_smooth = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-20]
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
gs_nb = GridSearchCV(sys_nb, param_grid={
                     "var_smoothing": val_smooth}, scoring="accuracy", cv=folds, verbose=1, n_jobs=-1)

gs_nb_trained = gs_nb.fit(X_train, y_train)

print("Accuracy para cada valor durante la búsqueda: ",
      gs_nb_trained.cv_results_["mean_test_score"])
print("Mejor combinación de hiperparámetros: ", gs_nb_trained.best_params_)
print("Mejor rendimiento obtenido en GridSearch: %0.4f" %
      gs_nb_trained.best_score_)
y_pred = gs_nb_trained.predict(X_test)
print("Accuracy en conjunto de test: %0.4f" %
      metrics.accuracy_score(y_test, y_pred))

print("###################################################################")
print("5. Realiza un calibrado de probabilidades y muestra las probabilidades obtenidas para los 5 primeros ejemplos")
print("###################################################################")

sys_nb_calib = CalibratedClassifierCV(
    gs_nb_trained, method="sigmoid", cv=folds)
# se entrena y se evalúa
sys_nb_calib.fit(X_train, y_train)
y_pred = sys_nb_calib.predict(X_test)
print("Accuracy: %.4f" % metrics.accuracy_score(y_test, y_pred))

# probabilidades
cuantos = 10
print("Probabilidades de los %d primeros ejemplos" % cuantos)
print(sys_nb_calib.predict_proba(X_test.iloc[0:cuantos]))
print("Orden de las clases:", sys_nb_calib.classes_)
