print("###################################################################")
print(
    "1. Para realizar la carga debes tener en cuenta si tiene o no valores desconocidos y si no tiene cabecera debes asignar nombres a las columnas mediante el parámetro 'names'"
)
print("###################################################################")

import pandas as pd
from sklearn import preprocessing, impute

cabecera = ["BI-RADS", "Age", "Shape", "Margin", "Density", "class"]
df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=cabecera,
    na_values="?",
)

print(df.head(5))
print(df.describe())


print("###################################################################")
print(
    "2. Haz un escalado en el rango [0,1] (MinMaxScaler) y compara los datos antes y despues del escalado. Aplica el escalado a todas las columnas menos a la clase"
)
print("###################################################################")

scaler = preprocessing.MinMaxScaler()
atributos = ["BI-RADS", "Age", "Shape", "Margin", "Density"]
df[atributos] = scaler.fit_transform(df[atributos])

print(df.head(5))
print(df.describe())

print("###################################################################")
print(
    "3. Vuelve a cargar el fichero y repite el apartado 2 realizando una estandarización (StandardScaler)"
)
print("###################################################################")

df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=cabecera,
    na_values="?",
)

scaler = preprocessing.StandardScaler()
df[atributos] = scaler.fit_transform(df[atributos])

print(df.head(5))
print(df.describe())

print("###################################################################")
print(
    "4. Carga de nuevo el fichero, elimina los ejemplos con valores desconocidos y compara el conjunto antes y después de la eliminación. (dropna)"
)
print("###################################################################")

df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=cabecera,
    na_values="?",
)

print(df.head(5))
print(df.describe())

df = df.dropna()

print(df.head(5))
print(df.describe())

print("###################################################################")
print(
    "5. Repite el apartado 2 pero en lugar de eliminar los desconocidos trata de asignarles valores. (simpleimputer, knnimputer)"
)
print("###################################################################")

print("\n Por SimpleImputer (Mean). \n")

df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=cabecera,
    na_values="?",
)

print(df.head(5))
print(df.describe())

imputer_media = impute.SimpleImputer(strategy="mean")
df[atributos] = imputer_media.fit_transform(df[atributos])

print(df.head(5))
print(df.describe())

#####
print("\n Por KNNImputer. \n")

df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=cabecera,
    na_values="?",
)

print(df.head(5))
print(df.describe())

imputer_media = impute.KNNImputer()
df[atributos] = imputer_media.fit_transform(df[atributos])

print(df.head(5))
print(df.describe())
