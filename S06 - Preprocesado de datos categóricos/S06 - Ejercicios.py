print(
    "1. Carga el fichero **breast-cancer.data** (es un archivo de texto). Para realizar la carga debes tener en cuenta si tiene o no valores desconocidos y si no tiene cabecera debes asignar nombres a las columnas mediante el parámetro 'names'"
)

import pandas as pd
from sklearn import preprocessing, impute

header = [
    "class",
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
    names=header,
    na_values="?",
)

print("3. Trata los valores desconocidos")

imputer_media = impute.SimpleImputer(strategy="most_frequent")
df[df.columns] = imputer_media.fit_transform(df[df.columns])

print(
    "4. Convierte los atributos categóricos nominales como se ha explicado en el notebook"
)

one_hot = preprocessing.LabelBinarizer()
df[one_hot.classes_] = one_hot.fit_transform(df["breast-quad"])
df = df.drop("breast-quad", axis=1)

label_enc = preprocessing.LabelEncoder()
df["node-caps"] = label_enc.fit_transform(df["node-caps"])
df["breast"] = label_enc.fit_transform(df["breast"])
df["irradiat"] = label_enc.fit_transform(df["irradiat"])

print(df.head(5))


print(
    "5. Convierte los atributos categóricos ordinales como se ha explicado en el notebook"
)

mapeo = {
    "10-19": 1,
    "20-29": 2,
    "30-39": 3,
    "40-49": 4,
    "50-59": 5,
    "60-69": 6,
    "70-79": 7,
    "80-89": 8,
    "90-99": 9,
}
# Age
df["age"] = df["age"].replace(mapeo)

# Men.
mapeo = {"lt40": 1, "ge40": 2, "premeno": 3}
df["menopause"] = df["menopause"].replace(mapeo)

# Tum. Size

# Inv. nodes.

print("6. Convierte la clase")
df["class"] = label_enc.fit_transform(df["class"])


print(df.head(15))
