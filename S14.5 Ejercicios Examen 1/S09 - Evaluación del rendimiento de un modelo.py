import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    LeaveOneOut,
    StratifiedKFold,
)
import matplotlib.pyplot as plt

print("###################################################################")
print(
    "1. Carga el fichero **heart_failure_clinical_records_dataset.csv** (es un archivo de texto). "
)
print("###################################################################")

df = pd.read_csv(
    "S09 - Evaluación de un modelo/heart_failure_clinical_records_dataset.csv", header=0
)
filas, columnas = df.shape

X = df.iloc[:, 0 : (columnas - 1)]
print(X)
y = df.iloc[:, (columnas - 1)]
print(y)

print("###################################################################")
print("2. Crea el baseline de la clase mayoritaria y un KNN con 3 vecinos")
print("###################################################################")

cl_my_sis = DummyClassifier(strategy="most_frequent")
knn_sis = KNeighborsClassifier(n_neighbors=3)

print("###################################################################")
print(
    "3. Haz un hold-out con una partición 75-25 (ten en cuenta que los atributos tienen escalas diferentes) --> ESTANDARIZAR."
)
print("###################################################################")

# Hold-out 75-25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1234, test_size=0.25
)

std_sca = StandardScaler()
std_sca.fit(X_train)

# Transformar el conjunto de entrenamiento y el de test
X_train_std = std_sca.transform(X_train)
X_test_std = std_sca.transform(X_test)

cl_my_sis.fit(X_train_std, y_train)
y_pred = cl_my_sis.predict(X_test_std)
print("Accuracy clase mayoritaria: %.3f" % metrics.accuracy_score(y_test, y_pred))

knn_sis.fit(X_train_std, y_train)
y_pred = knn_sis.predict(X_test_std)
print("Accuracy KNN con 3 vecinos: %.3f" % metrics.accuracy_score(y_test, y_pred))


print("###################################################################")
print("4. Haz una validación cruzada de 10 folds.")
print("###################################################################")

# Generador de folds estratificados partiendo el conjunto.
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

# Validación cruzada para el baseline
scores_cl_may = cross_val_score(cl_my_sis, X, y, cv=folds, scoring="accuracy")

# Validación cruzada para KNN - Estandarizando.
std_knn = Pipeline([("std", std_sca), ("knn", knn_sis)])
scores_std_knn = cross_val_score(std_knn, X, y, cv=folds, scoring="accuracy")

print(
    "Clase mayoritaria    (mean +- std): %0.4f +- %0.4f"
    % (scores_cl_may.mean(), scores_cl_may.std())
)
print(
    "STD_KNN con 3 vecinos    (mean +- std): %0.4f +- %0.4f"
    % (scores_std_knn.mean(), scores_std_knn.std())
)

print("###################################################################")
print("5. Haz un leave-one-out.")
print("###################################################################")

# Validación cruzada para el baseline
scores_cl_may = cross_val_score(cl_my_sis, X, y, cv=LeaveOneOut(), scoring="accuracy")

# Validación cruzada para KNN - Estandarizando.
std_knn = Pipeline([("std", std_sca), ("knn", knn_sis)])
scores_std_knn = cross_val_score(std_knn, X, y, cv=LeaveOneOut(), scoring="accuracy")

print(
    "Clase mayoritaria    (mean +- std): %0.4f +- %0.4f"
    % (scores_cl_may.mean(), scores_cl_may.std())
)
print(
    "STD_KNN con 3 vecinos    (mean +- std): %0.4f +- %0.4f"
    % (scores_std_knn.mean(), scores_std_knn.std())
)

print("###################################################################")
print(
    "6. Haz una gráfica en la que se vea la evolución (en función del número de vecinos [1..10]) de la accuracy obtenida en una validación cruzada de 10 folds estandarizando y sin estandarizar."
)
print("###################################################################")

# Comparar estandarizar vs no estandarizar en un a 10FCV

list_n_neighbors = range(1, 11)
knn_sis = KNeighborsClassifier()

print("Sin estandarizar: ")
no_std = []
for i in list_n_neighbors:
    knn_sis.set_params(n_neighbors=i)
    scores_knn = cross_val_score(knn_sis, X, y, cv=folds, scoring="accuracy")
    acc = scores_knn.mean()
    no_std.append(acc)
    print("N_neighbors = %d : %.3f" % (i, acc))

print("Estandarizando: ")
si_std = []
for i in list_n_neighbors:
    std_knn.set_params(knn__n_neighbors=i)
    scores_std_knn = cross_val_score(std_knn, X, y, cv=folds, scoring="accuracy")
    acc = scores_std_knn.mean()
    si_std.append(acc)
    print("N_neighbors = %d : %.3f" % (i, acc))

plt.plot(range(1, 11), no_std, label="Sin estandarizar")
plt.plot(range(1, 11), si_std, label="Con estandarización")
plt.title("Accuracy vs número de vecinos")
plt.legend()
plt.show()
