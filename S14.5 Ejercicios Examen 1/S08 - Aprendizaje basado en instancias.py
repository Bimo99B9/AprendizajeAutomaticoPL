import pandas as pd
from sklearn import impute, metrics
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
print(
    "2. Convierte los atributos categóricos y asigna valores a los missing si los hay."
)
print("###################################################################")

# Transformar el atributo "gender", es binario.
lb_enc = LabelEncoder()
X["Gender"] = lb_enc.fit_transform(X["Gender"])
print(X.head(5))

# Asignar missing values
print(X.describe())
imputer_knn = impute.KNNImputer(n_neighbors=2)
X[X.columns] = imputer_knn.fit_transform(X)
print(X.describe())


print("###################################################################")
print("3. Calcula el error con el baseline de 'la clase más frecuente'.")
print("###################################################################")

sis = DummyClassifier(strategy="most_frequent")
sis.fit(X, y)
y_pred = sis.predict(X)
print("Resultados para DummyClassifier con strategy='most_frequent'")
print("Accuracy: %.4f" % metrics.accuracy_score(y, y_pred))
print(
    "(P=%.2f, R=%.2f, F1=%.2f)"
    % (
        metrics.precision_score(y, y_pred),
        metrics.recall_score(y, y_pred),
        metrics.f1_score(y, y_pred),
    )
)


print("###################################################################")
print(
    "4. Calcula la accuracy para un K-vecinos (KNN) variando el número de vecinos desde 1 hasta 10."
)
print("###################################################################")

knn_sis = KNeighborsClassifier()

no_std = []
for i in range(1, 11):
    knn_sis.set_params(n_neighbors=i)
    knn_sis.fit(X, y)
    y_pred = knn_sis.predict(X)
    acc = metrics.accuracy_score(y, y_pred)
    no_std.append(acc)
    print(
        "N_Neighbors = %d: (P=%.3f, R=%.3f, F1=%.3f)"
        % (
            i,
            metrics.precision_score(y, y_pred),
            metrics.recall_score(y, y_pred),
            metrics.f1_score(y, y_pred),
        )
    )


print("###################################################################")
print(
    "5. Crea un pipeline estandarizando y con un k-vecinos y calcula la accuracy variando el número de vecinos desde 1 hasta 10."
)
print("###################################################################")

std_knn = Pipeline([("std", StandardScaler()), ("knn", KNeighborsClassifier())])
si_std = []
for i in range(1, 11):
    std_knn.set_params(knn__n_neighbors=i)
    std_knn.fit(X, y)
    y_pred = std_knn.predict(X)
    acc = metrics.accuracy_score(y, y_pred)
    si_std.append(acc)
    print(
        "N_Neighbors = %d: (P=%.3f, R=%.3f, F1=%.3f)"
        % (
            i,
            metrics.precision_score(y, y_pred),
            metrics.recall_score(y, y_pred),
            metrics.f1_score(y, y_pred),
        )
    )

print("###################################################################")
print(
    "6. Representa en una gráfica la evolución de la accuracy en función del número de vecinos."
)
print("###################################################################")

plt.plot(range(1, 11), no_std, label="Sin estandarizar")
plt.plot(range(1, 11), si_std, label="Con estandarización")
plt.title("Accuracy vs número de vecinos")
plt.legend()
plt.show()
