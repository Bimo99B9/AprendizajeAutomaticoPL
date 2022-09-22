from sklearn import preprocessing, impute
import pandas as pd

df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"],
    na_values="?",
)

print(df.head(5))
print(df.describe())

print(
    "2. Haz un escalado en el rango [0,1] y compara los datos antes y despues del escalado. Aplica el escalado a todas las columnas menos a la clase"
)

scaler = preprocessing.MinMaxScaler()
features = ["BI-RADS", "Age", "Shape", "Margin", "Density"]
df[features] = scaler.fit_transform(df[features])
print(df.head(5))
print(df.describe())

print(
    "3. Vuelve a cargar el fichero y repite el apartado 2 realizando una estandarización"
)

df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"],
    na_values="?",
)

normalizer = preprocessing.StandardScaler()
df[features] = scaler.fit_transform(df[features])
print(df.head(5))
print(df.describe())


print(
    "4. Carga de nuevo el fichero, elimina los ejemplos con valores desconocidos y compara el conjunto antes y después de la eliminación."
)

df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"],
    na_values="?",
)

df = df.dropna()
print(df.describe())

print(
    "5. Repite el apartado 2 pero en lugar de eliminar los desconocidos trata de asignarles valores."
)

df = pd.read_csv(
    "S05 - Preprocesado de datos númericos/mammographic_masses.data",
    names=["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"],
    na_values="?",
)

imputer_knn = impute.KNNImputer()
df[features] = imputer_knn.fit_transform(df[features])
print(df.describe())
