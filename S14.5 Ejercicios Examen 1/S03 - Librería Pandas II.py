print("###################################################################")
print("1. Agrupa por paciente ('Observation_number') y calcula los valores medios")
print("###################################################################")

import pandas as pd

df = pd.read_csv("S03 - Librería Pandas II/biomed.data", header=0)
print(df.head(5))

print(df.groupby("Observation_number").mean())

print("###################################################################")
print("2. Agrupa por paciente y clase y calcula la media de 'm1'")
print("###################################################################")

print(df.groupby(["Observation_number", "class"])["m1"].mean())

print("###################################################################")
print(
    "3. Agrupa por paciente y clase y calcula la media y desviación de la edad y el máximo y el mínimo de m4"
)
print("###################################################################")

print(
    df.groupby(["Observation_number", "class"]).agg(
        {"Age_of_patient": ["mean", "std"], "m4": ["max", "min"]}
    )
)

print("###################################################################")
print(
    "4. Cambia la edad a meses y el nombre de la columna a 'Months' (busca ayuda de la función `rename()` para renombrar una columna)"
)
print("###################################################################")

df["Age_of_patient"] = df["Age_of_patient"].apply(lambda x: x * 12)
df = df.rename(columns={"Age_of_patient": "Months"})
print(df.head(10))

print("###################################################################")
print(
    "5. Haz un filtro para ver las filas en las que 'm3' sea menor que 10 o mayor que 100 y que la clase NO sea normal"
)
print("###################################################################")

print(df[((df["m3"] < 10) | (df["m3"] > 100)) & ~(df["class"] == "normal")])
