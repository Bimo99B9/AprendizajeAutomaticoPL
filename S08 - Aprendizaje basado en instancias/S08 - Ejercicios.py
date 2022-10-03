
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import impute, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


print("#######################################################")
print("1. Carga el fichero **ILPD.data** (es un archivo de texto).  ")
print("#######################################################")

header = [
    "age",
    "gender",
    "tb",
    "db",
    "alkphos",
    "sgpt",
    "sgot",
    "tp",
    "alb",
    "ag",
    "class"
]

df = pd.read_csv('S08 - Aprendizaje basado en instancias\ILPD.data', names=header)

print(df.head(3))

print("#######################################################")
print("2. Convierte los atributos categóricos y asigna valores a los missing si los hay.")
print("#######################################################")

class_enc = preprocessing.LabelEncoder()

df['gender'] = class_enc.fit_transform(df['gender'])

imputer_media = impute.SimpleImputer(strategy='most_frequent')

df[df.columns] = imputer_media.fit_transform(df[df.columns])

print(df.head(5))

print("#######################################################")
print("3. Calcula el error con el baseline de 'la clase más frecuente'.")
print("#######################################################")

from sklearn.dummy import DummyClassifier
from sklearn import metrics

X = df.drop(["class"], axis=1)
y = df["class"]

sis = DummyClassifier(strategy="most_frequent", random_state=9949)
sis.fit(X, y)
y_pred = sis.predict(X)
print("Estrategia 'most frequent'")
print("Accuracy: ", metrics.accuracy_score(y, y_pred))

print("#######################################################")
print("4. Calcula la accuracy para un K-vecinos (KNN) variando el número de vecinos desde 1 hasta 10.")
print("#######################################################")

for num_vecinos in range(1, 10):
    print(f"Número de vecinos: {num_vecinos}")
    knn_sis = KNeighborsClassifier(n_neighbors=num_vecinos)
    knn_sis.fit(X, y)
    y_pred = knn_sis.predict(X)
    print("Accuracy: ", metrics.accuracy_score(y, y_pred))


print("#######################################################")
print("5. Crea un pipeline estandarizando y con un k-vecinos y calcula la accuracy variando el número de vecinos desde 1 hasta 10.")
print("#######################################################")

std_knn = Pipeline([('std', preprocessing.StandardScaler()), ('knn', KNeighborsClassifier())])

accuracies = []
for num_vecinos in range(1, 10):
    std_knn.set_params(knn__n_neighbors=num_vecinos)
    std_knn.fit(X, y)
    y_pred = std_knn.predict(X)
    acc = metrics.accuracy_score(y, y_pred)
    print("Accuracy: ", acc)
    accuracies.append(acc)


print("#######################################################")
print("6. Representa en una gráfica la evolución de la accuracy en función del número de vecinos.")
print("#######################################################")

fig, ax = plt.subplots()

ax.scatter(x=list(range(1,10)), y = accuracies)
ax.set_title('Accuracy en función del número de vecinos')
ax.set_xlabel('Número de vecinos')
ax.set_ylabel('Accuracy') 
plt.show()
