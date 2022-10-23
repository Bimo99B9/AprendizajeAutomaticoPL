import pandas as pd
from sklearn import preprocessing, impute, metrics
from sklearn.dummy import DummyClassifier
import seaborn as sns
import matplotlib.pyplot as plt

print("###################################################################")
print("1. Carga el fichero **ILPD.data** (es un archivo de texto).")
print("###################################################################")

cabecera = [
    "Age",
    "Gender",
    "TB",
    "DB",
    "Alkphos",
    "Sgpt",
    "Sgot",
    "TP",
    "ALB",
    "A/G",
    "Selector (class)",
]
df = pd.read_csv(
    "S07 - Calcular métricas en problemas de clasificación/ILPD.data", names=cabecera
)
filas, columnas = df.shape

X = df.iloc[:, 0 : (columnas - 1)]
print(X)

y = df.iloc[:, (columnas - 1)]
print(y)

print("###################################################################")
print("2. Cuenta los ejemplos que hay de cada clase")
print("###################################################################")

print(y.value_counts())

print("###################################################################")
print("3. Crea tres baselines y calcula accuracy, precision, recall y F1 para los tres")
print("###################################################################")

# Estrategia uniform
sis = DummyClassifier(strategy="uniform", random_state=1234)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\n Resultados para 'uniform'")
print("Accuracy  : %.4f" % metrics.accuracy_score(y, y_pred))
print("Precision : %.4f" % metrics.precision_score(y, y_pred))
print("Recall    : %.4f" % metrics.recall_score(y, y_pred))
print("F1        : %.4f" % metrics.f1_score(y, y_pred))

# Estrategia stratified
sis = DummyClassifier(strategy="stratified", random_state=1234)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\n Resultados para 'stratified'")
print("Accuracy  : %.4f" % metrics.accuracy_score(y, y_pred))
print("Precision : %.4f" % metrics.precision_score(y, y_pred))
print("Recall    : %.4f" % metrics.recall_score(y, y_pred))
print("F1        : %.4f" % metrics.f1_score(y, y_pred))

# Estrategia most frequent
sis = DummyClassifier(strategy="most_frequent", random_state=1234)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\n Resultados para 'most frequent'")
print("Accuracy  : %.4f" % metrics.accuracy_score(y, y_pred))
print("Precision : %.4f" % metrics.precision_score(y, y_pred))
print("Recall    : %.4f" % metrics.recall_score(y, y_pred))
print("F1        : %.4f" % metrics.f1_score(y, y_pred))

print("###################################################################")
print("4. Repite los cálculos marcando como clase positiva la '0'")
print("###################################################################")

# Estrategia uniform
sis = DummyClassifier(strategy="uniform", random_state=1234)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\n Resultados para 'uniform'")
print("Accuracy  : %.4f" % metrics.accuracy_score(y, y_pred))
print("Precision : %.4f" % metrics.precision_score(y, y_pred, pos_label=0))
print("Recall    : %.4f" % metrics.recall_score(y, y_pred, pos_label=0))
print("F1        : %.4f" % metrics.f1_score(y, y_pred, pos_label=0))

# Estrategia stratified
sis = DummyClassifier(strategy="stratified", random_state=1234)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\n Resultados para 'stratified'")
print("Accuracy  : %.4f" % metrics.accuracy_score(y, y_pred))
print("Precision : %.4f" % metrics.precision_score(y, y_pred, pos_label=0))
print("Recall    : %.4f" % metrics.recall_score(y, y_pred, pos_label=0))
print("F1        : %.4f" % metrics.f1_score(y, y_pred, pos_label=0))

# Estrategia most frequent
sis = DummyClassifier(strategy="most_frequent", random_state=1234)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\n Resultados para 'most frequent'")
print("Accuracy  : %.4f" % metrics.accuracy_score(y, y_pred))
print("Precision : %.4f" % metrics.precision_score(y, y_pred, pos_label=0))
print("Recall    : %.4f" % metrics.recall_score(y, y_pred, pos_label=0))
print("F1        : %.4f" % metrics.f1_score(y, y_pred, pos_label=0))


print("###################################################################")
print("5. Calcula la matriz de confusión de uno de los sistemas")
print("###################################################################")

# Estrategia stratified
sis = DummyClassifier(strategy="stratified", random_state=1234)
sis.fit(X, y)
y_pred = sis.predict(X)

# Calculamos la matriz de confusión
class_labels = df["Selector (class)"].unique()
print("Clases: ", class_labels)

cm = metrics.confusion_matrix(y, y_pred, labels=class_labels)
print("Matriz de confusión: ")
print(cm)
