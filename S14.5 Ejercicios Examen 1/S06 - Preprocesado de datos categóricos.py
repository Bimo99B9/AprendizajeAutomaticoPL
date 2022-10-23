print("###################################################################")
print("1. Carga el fichero **breast-cancer.data** (es un archivo de texto).")
print("###################################################################")

import pandas as pd
from sklearn import preprocessing, impute

cabecera = [
    "Class",
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
]

df = pd.read_csv(
    "S06 - Preprocesado de datos categóricos/breast-cancer.data",
    names=cabecera,
    na_values="?",
)

print(df.head(5))

print("###################################################################")
print("3. Trata los valores desconocidos")
print("###################################################################")

# Máscara que identifica las filas que tienen al menos un True
mask = df.isnull().any(axis=1)

# Ejemplos sin corregir.
print(df[mask])

imputer_media = impute.SimpleImputer(strategy="most_frequent")
df[df.columns] = imputer_media.fit_transform(df[df.columns])

# Ejemplos corregidos.
print(df[mask])

print("###################################################################")
print("4. Convierte los atributos categóricos nominales")
print("###################################################################")

# Para atributos categóricos con más de 2 clases (no binarios).
one_hot = preprocessing.LabelBinarizer()
df[one_hot.classes_] = one_hot.fit_transform(df["breast-quad"])
df = df.drop("breast-quad", axis=1)

# Para categóricos binarios
label_enc = preprocessing.LabelEncoder()

df["node-caps"] = label_enc.fit_transform(df["node-caps"])
df["breast"] = label_enc.fit_transform(df["breast"])
df["irradiat"] = label_enc.fit_transform(df["irradiat"])

print(df.head(10))

print("###################################################################")
print("5. Convierte los atributos categóricos ordinales")
print("###################################################################")

mapeo = {
    "10:19": 1,
    "20-29": 2,
    "30-39": 3,
    "40-49": 4,
    "50-59": 5,
    "60-69": 6,
    "70-79": 7,
    "80-89": 8,
    "90-99": 9,
}


df["age"] = df["age"].replace(mapeo)

mapeo = {"lt40": 1, "ge40": 2, "premeno": 3}
df["menopause"] = df["menopause"].replace(mapeo)

mapeo = {
    "0-4": 1,
    "5-9": 2,
    "10-14": 3,
    "15-19": 4,
    "20-24": 5,
    "25-29": 6,
    "30-34": 7,
    "35-39": 8,
    "40-44": 9,
    "45-49": 10,
    "50-54": 11,
    "55-59": 12,
}
df["tumor-size"] = df["tumor-size"].replace(mapeo)

# Falta inv-nodes, que es lo mismo.

print("###################################################################")
print("6. Convierte la clase")
print("###################################################################")

df["Class"] = label_enc.fit_transform(df["Class"])
print(df.head(5))

# Poner la clase al final.

colum_clase = 0

columnas = (
    df.columns[(colum_clase + 1) :].to_list()
    + df.columns[colum_clase : (colum_clase + 1)].to_list()
)

print(columnas)

df = df.reindex(columns=columnas)

print(df.head(5))
