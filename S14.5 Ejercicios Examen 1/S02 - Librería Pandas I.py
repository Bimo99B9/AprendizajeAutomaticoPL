print("###################################################################")
print("1. Muestra las 11 primeras filas.")
print("###################################################################")

import pandas as pd

df = pd.read_csv("S02 - Librería Pandas I/biomed.data", header=0)
print(df.head(11))

print("###################################################################")
print(
    "2. Utiliza el parámetro `na_values` de `read_csv()` para que en la lectura del fichero ya se tenga en cuenta que los valores desconocido vienen indicados con '?' (este mismo parámetro existe en `read_excel()`)."
)
print("###################################################################")

df = pd.read_csv("S02 - Librería Pandas I/biomed.data", header=0, na_values="?")
print(df.head(11))

print("###################################################################")
print(
    "3. Muestra los datos que hay entre las filas 45 y 50 para los atributos 'Hospital', 'Age_of_patient' y 'Date' (lo hago con iloc y a ver si se puede con loc)"
)
print("###################################################################")

print(df.iloc[45:51, 1:4])
print(df.loc[45:50, ["Hospital", "Age_of_patient", "Date"]])

print("###################################################################")
print("4. Muestra el número de datos que hay en cada columna")
print("###################################################################")

print(df.count())

print("###################################################################")
print("5. Elimina las columnas 'Observation_number' y 'Hospital'")
print("###################################################################")

df = df.drop(["Observation_number", "Hospital"], axis=1)
print(df.head(5))

print("###################################################################")
print("6. Analiza la columna categórica 'class'")
print("###################################################################")

print("\nNúmero de categorías: ", df["class"].nunique())
print("\nCategorías: ", df["class"].unique())
print("\nEjemplos por categoría: ")
print(df["class"].value_counts())
