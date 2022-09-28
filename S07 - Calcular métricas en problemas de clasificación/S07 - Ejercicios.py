import pandas as pd

print("#######################################################")
print("1. Carga el fichero **ILPD.data** (es un archivo de texto). ")
print("#######################################################")

cabecera = [
    "age",
    "gender",
    "tb",
    "db",
    "alphos",
    "sgpt",
    "sgot",
    "tp",
    "alb",
    "ag ratio",
    "selector",
]

df = pd.read_csv(
    "S07 - Calcular métricas en problemas de clasificación\ILPD.data", names=cabecera
)
print(df.head(5))

print("#######################################################")
print("2. Cuenta los ejemplos que hay de cada clase ")
print("#######################################################")

print(df["selector"].value_counts())
print(df["selector"].value_counts(normalize=True))


print("#######################################################")
print("3. Crea tres baselines y calcula accuracy, precision, recall y F1 para los tres")
print("#######################################################")

from sklearn.dummy import DummyClassifier
from sklearn import metrics

X = df.drop(["selector"], axis=1)
y = df["selector"]

sis = DummyClassifier(strategy="uniform", random_state=9949)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\nEstrategia 'uniform'")
print("Accuracy  :", metrics.accuracy_score(y, y_pred))
print("Precision :", metrics.precision_score(y, y_pred))
print("Recall    :", metrics.recall_score(y, y_pred))
print("F1        :", metrics.f1_score(y, y_pred))

sis = DummyClassifier(strategy="stratified", random_state=9949)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\nEstrategia 'stratisfied'")
print("Accuracy  :", metrics.accuracy_score(y, y_pred))
print("Precision :", metrics.precision_score(y, y_pred))
print("Recall    :", metrics.recall_score(y, y_pred))
print("F1        :", metrics.f1_score(y, y_pred))

sis = DummyClassifier(strategy="most_frequent", random_state=9949)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\nEstrategia 'most frequent'")
print("Accuracy  :", metrics.accuracy_score(y, y_pred))
print("Precision :", metrics.precision_score(y, y_pred))
print("Recall    :", metrics.recall_score(y, y_pred))
print("F1        :", metrics.f1_score(y, y_pred))

print("#######################################################")
print("4. Repite los cálculos marcando como clase positiva la '0'")
print("#######################################################")

sis = DummyClassifier(strategy="uniform", random_state=9949)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\nEstrategia 'uniform'")
# print("Accuracy  :", metrics.accuracy_score(y, y_pred, pos_label=0))
print("Precision :", metrics.precision_score(y, y_pred, pos_label=0))
print("Recall    :", metrics.recall_score(y, y_pred, pos_label=0))
print("F1        :", metrics.f1_score(y, y_pred, pos_label=0))

sis = DummyClassifier(strategy="stratified", random_state=9949)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\nEstrategia 'stratisfied'")
# print("Accuracy  :", metrics.accuracy_score(y, y_pred, pos_label=0))
print("Precision :", metrics.precision_score(y, y_pred, pos_label=0))
print("Recall    :", metrics.recall_score(y, y_pred, pos_label=0))
print("F1        :", metrics.f1_score(y, y_pred, pos_label=0))

sis = DummyClassifier(strategy="most_frequent", random_state=9949)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\nEstrategia 'most frequent'")
# print("Accuracy  :", metrics.accuracy_score(y, y_pred, pos_label=0))
print("Precision :", metrics.precision_score(y, y_pred, pos_label=0))
print("Recall    :", metrics.recall_score(y, y_pred, pos_label=0))
print("F1        :", metrics.f1_score(y, y_pred, pos_label=0))

print("#######################################################")
print("5. Calcula la matriz de confusión de uno de los sistemas")
print("#######################################################")

sis = DummyClassifier(strategy="uniform", random_state=9949)
sis.fit(X, y)
y_pred = sis.predict(X)
print("\nEstrategia 'uniform'")
print("Precision :", metrics.precision_score(y, y_pred, pos_label=0))
print("Recall    :", metrics.recall_score(y, y_pred, pos_label=0))
print("F1        :", metrics.f1_score(y, y_pred, pos_label=0))

cm = metrics.confusion_matrix(y, y_pred)
print("Matriz de confusión:")
print(cm)
