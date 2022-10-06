import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

print("#######################################################")
print(
    "1. Carga el fichero **heart_failure_clinical_records_dataset.csv** (es un archivo de texto)."
)
print("#######################################################")

df = pd.read_csv(
    "S09 - Evaluación de un modelo/heart_failure_clinical_records_dataset.csv"
)
print(df.head(5))

X = df.drop(["DEATH_EVENT"], axis=1)
y = df["DEATH_EVENT"]


print("#######################################################")
print("2. Crea el baseline de la clase mayoritaria y un KNN con 3 vecinos")
print("#######################################################")

cl_my_sis = DummyClassifier(strategy="most_frequent")
knn_sis = KNeighborsClassifier(n_neighbors=3)

print("#######################################################")
print(
    "3. Haz un hold-out con una partición 75% - 25\% (ten en cuenta que los atributos tienen escalas diferentes)."
)
print("#######################################################")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=9949, stratify=y
)


std_sca = preprocessing.StandardScaler()

std_sca.fit(X_train)


X_train_std = std_sca.transform(X_train)
X_test_std = std_sca.transform(X_test)

## Baseline clase mayoritaria
cl_my_sis.fit(X_train_std, y_train)
y_pred = cl_my_sis.predict(X_test_std)
print(f"Accuracy clase mayoritaria: {accuracy_score(y_test, y_pred)}")
##

## KNN con 3 vecinos.
knn_sis.fit(X_train_std, y_train)
y_pred = knn_sis.predict(X_test_std)
print(f"Accuracy KNN con 3 vecinos: {accuracy_score(y_test, y_pred)}")
##


print("#######################################################")
print("4. Haz una validación cruzada de 10 folds.")
print("#######################################################")

# Validación cruzada para el baseline.
scores_cl_may = cross_val_score(cl_my_sis, X, y, cv=10, scoring="accuracy")

# Standarized KNN, validación cruzada.
std_knn = Pipeline([("std", std_sca), ("knn", knn_sis)])
scores_std_knn = cross_val_score(std_knn, X, y, cv=10, scoring="accuracy")

print(f"Baseline: {scores_cl_may.mean()} +- {scores_cl_may.std()}")
print(f"STDKNN: {scores_std_knn.mean()} +- {scores_std_knn.std()}")

print("#######################################################")
print("5. Haz un leave-one-out.")
print("#######################################################")

# Leave-one-out para el baseline.
scores_cl_may = cross_val_score(cl_my_sis, X, y, cv=LeaveOneOut(), scoring="accuracy")

# Standarized KNN, validación cruzada.
std_knn = Pipeline([("std", std_sca), ("knn", knn_sis)])
scores_std_knn = cross_val_score(std_knn, X, y, cv=LeaveOneOut(), scoring="accuracy")

print(f"Baseline: {scores_cl_may.mean()} +- {scores_cl_may.std()}")
print(f"STDKNN: {scores_std_knn.mean()} +- {scores_std_knn.std()}")


print("#######################################################")
print(
    "6. Haz una gráfica en la que se vea la evolución (en función del número de vecinos [1..10]) de la accuracy obtenida en una validación cruzada de 10 folds estandarizando y sin estandarizar."
)
print("#######################################################")

allscores_knn = []
allscores_std_knn = []

knn_sis = KNeighborsClassifier()

for num_vecinos in range(1, 11):

    knn_sis.set_params(n_neighbors=num_vecinos)

    scores_knn = cross_val_score(knn_sis, X, y, cv=10, scoring="accuracy")
    allscores_knn.append(scores_knn.mean())

    std_knn = Pipeline([("std", std_sca), ("knn", knn_sis)])
    std_knn.set_params(knn__n_neighbors=num_vecinos)
    scores_std_knn = cross_val_score(std_knn, X, y, cv=10, scoring="accuracy")
    allscores_std_knn.append(scores_std_knn.mean())

plt.plot(list(range(1, 11)), allscores_knn, label="No std")
plt.plot(list(range(1, 11)), allscores_std_knn, label="Std")
plt.title("Accuracy vs número de vecinos")
plt.legend()
plt.show()
