#1. Muestra las 11 primeras filas. Verás que el último ejemplo que se muestra tiene '?' en 'm2'. Esto es debido a que los *missing values* suelen señalarse de diferentes maneras y una de ellas es escribir un '?'.

import pandas as pd

df = pd.read_csv('S02 - Librería Pandas I/biomed.data', sep=',')
print(df.head(11))

#2. Utiliza el parámetro `na_values` de `read_csv()` para que en la lectura del fichero ya se tenga en cuenta que los valores desconocido vienen indicados con '?' (este mismo parámetro existe en `read_excel()`).

df = pd.read_csv('S02 - Librería Pandas I/biomed.data', sep=',', na_values='?')
print(df.head(11))

#3. Datos entre las filas 45 y 50 para 'Hospital', 'Age_of_patient' y 'Date'.
print(df.iloc[45:51, 1:4])

#4. Muestra el número de datos que hay en cada columna.
print(df.count())

#5. Elimina las columnas 'Observation_number' y 'Hospital'
df = df.drop(['Observation_number', 'Hospital'], axis=1)
print(df.head(3))

#6. Analiza la columna categórica 'class'.
print(df['class'].unique())
print(df['class'].nunique())
print(df['class'].value_counts())
# Class is bool with 134 normal atts. and 75 carrier atts. int64 datatype.