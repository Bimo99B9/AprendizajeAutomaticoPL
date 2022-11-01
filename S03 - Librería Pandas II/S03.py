# %% [markdown]
# ![Logo de AA1](logo_AA1_texto_small.png)
# # Sesión 3 - Pandas II
# En esta práctica vamos a continuar viendo algunas funcionales interesantes de la librería `pandas` para el tratamiento y preparación de los datos mediante el uso de `DataFrames`.
#
# **IMPORTANTE**: las funciones que vamos a ver en esta práctica pueden utilizarse de diversas maneras y con diferentes parámetros, lo que hará que su comportamiento varíe. Esta práctica no pretende explicar todoas las funcionalidades de la librería `pandas` (se necesitarían muchas horas para ello) sino que pretende únicamente ser una introducción a algunas de las características más importantes que tiene esta librería de cara a su uso en el tratamiento de conjuntos de datos. Información más detallada podréis encontrar en <https://pandas.pydata.org/docs/index.html>
#
# Para ello vamos a trabajar con el conjunto de datos **Irish_certificate.xlsx**. Lo primero que haremos será cargar los datos como ya vimos en la sesión anterior:

# %%
# se importa la librería pandas nombrándola pd
import pandas as pd

# cargamos los datos de una hoja Excel
df = pd.read_excel("Irish_certificate.xlsx", sheet_name="Data", header=0)

print(df)

# %% [markdown]
# ## 3.1 Agrupando ejemplos en función de valores
# Vamos a ver una instrucción muy interesante que se llama `groupby()`.
#
# A veces nos puede resultar interesante agrupar todos los ejemplos que comparten una misma característica para realizar algún cálculo sobre los mismos de forma agregada.
#
# Por ejemplo, podría interesarnos agrupar a las personas por sexo para conocer la media del 'Prestige_score' en cada uno de los grupos. Se haría de la siguiente manera:

# %%
print('\n############## "Prestige_score" medio por sexo ##############')
print(df.groupby("Sex")["Prestige_score"].mean())

# %% [markdown]
# Lo que ha hecho la instrucción es, agrupar los ejemplos por sexo, separar la columna 'Prestige_score' de cada grupo y calcular su media. En lugar de la media podríamos haber usado cualquier función que realice un calculo sobre los valores contenidos en el grupo. Podríamos usa, por ejemplo, `max()`, `min()`, `std()`, `count()`, `sum()`,...
#
# Si lo que nos interesa el la media de todas las columnas de tipo numérico, entonces basta con no indicar una columna sobre la que hacer el cálculo. Si no le indicamos una columna, entonces realizará el cálculo sobre todas las columnas que pueda:

# %%
print("\n############## Cálculo de la media en las columnas por sexo ##############")
print(df.groupby("Sex").mean())

# %% [markdown]
# En este caso reliza el cálculo sobre las dos columnas numéricas que tiene el conjunto.
#
# Si queremos también podemos realizar agrupaciones atendiendo a dos variables y realizar cálculos sobre los grupos resultantes:

# %%
print("\n############## Agrupando por sexo y tipo de escuela ##############")
print(df.groupby(["Sex", "Type_school"]).mean())

# %% [markdown]
# En este caso hemos agrupado por sexo y tipo de escuela y han resultado 6 grupos. El resultado de la instrucción genera un `DataFrame`que podemos almacenar en otra variable para acceder a los datos que nos interesen:

# %%
res = df.groupby(["Sex", "Type_school"]).mean()
print(type(res))

# %% [markdown]
# Pero en `res`, ¿cuáles son los índices?. Ahora ya no nos aparecen numerados los ejemplos de 0 en adelante y eso puede despistarnos.
#
# La propiedad `axes` almacena en un array los índices de los ejes. En este caso hay 2 ejes y podemos acceder a los índices de cada uno de ellos:

# %%
print("\n############## Consultamos los índices de todos los ejes ##############")
print(res.axes)

print("\n############## Consultamos los índices del eje 0 (filas) ##############")
print(res.axes[0])

print("\n############## Consultamos los índices del eje 1 (columnas) ##############")
print(res.axes[1])

# %% [markdown]
# Vemos que son índices no numéricos, se utilizan etiquetas.
#
# En el caso de las filas vemos que tiene un índice compuesto y en el de las columnas un índice normal. Vamos a ver ahora cómo podríamos acceder a los elementos de este `DataFrame` mediante el uso de `loc[]` e `iloc[]`.

# %%
print("\n############## Accedemos a una fila ##############")
print(res.loc[("female", "vocational")])

print("\n############## Accedemos a un dato ##############")
print(res.loc[("male", "secondary"), "Prestige_score"])

# %% [markdown]
# Para acceder con `loc[]` utilizamos las etiquetas como índices. En el caso de las filas es un índice compuesto, así que debemos utilizar una tupla, `('female', 'vocational')`, para indicar la fila a la que queremos acceder.
#
# Si además le especificamos la columna entonces accederemos a un dato en concreto.
#
# Ya habíamos comentado en la práctica anterior que los `DataFrames` siempre mantienen unos índices numéricos (aunque sea implícitamente). Así, podríamos utilizar `iloc[]` junto con esos índices numéricos para acceder a los mismos datos que en el ejemplo anterior:

# %%
print("\n############## Accedemos a la misma fila ##############")
print(res.iloc[2])

print("\n############## Accedemos al mismo dato ##############")
print(res.iloc[4, 1])

# %% [markdown]
# Si queremos aplicar funciones diferentes dependiendo de la columna, entonces debemos utilizar `agg()` en combinación con `groupby()`:

# %%
print(
    df.groupby("Leaving_certificate").agg(
        {"Prestige_score": ["mean", "std"], "DVRT": "min"}
    )
)

# %% [markdown]
# ## 3.2 Aplicando una función a todos los elementos de una columna
#
# A veces puede resultar necesario aplicar una función a todos los elementos de una columna. En estos casos `pandas` cuenta con el método `apply()`, al cual podemos indicarle la función que queremos aplicar:

# %%
print("\n############## Definimos la función ##############")


def mayusculas(x):
    return x.upper()


print("\n############## Aplicamos la función ##############")
print(df["Sex"].apply(mayusculas))

# %%
print("\n############## Aplicamos la función ##############")
print(df["Sex"].apply(lambda x: x.upper()))

# %% [markdown]
# La instrucción anterior no modifica el `DataFrame` sino que se limita a generar el resultado. Si queremos que el cambio quede reflejado en `df` debemos asignar el resultado a la columna correspondiente:

# %%
print("\n############## Aplicamos la función ##############")
df["Sex"] = df["Sex"].apply(mayusculas)
print(df)

# %% [markdown]
# ## 3.3 Aplicando filtros
# Una forma de quedarnos con las filas que nos interesan es mediante la utilización de una máscara. Una máscara es un array de `boolean` que indica las filas seleccionadas. Si le pasamos la máscara al `DataFrame` (`df[máscara]`) entonces nos devolverá únicamente las filas para las que el array tiene almacenado `True`.

# %%
print('\n############## Personas con "Prestige_score" inferior a 20 ##############')
mask = df["Prestige_score"] < 20
print(df[mask])

# %% [markdown]
# Estas instrucciones se suelen escribir de forma más compacta en una sola línea:

# %%
print('\n############## Personas con "Prestige_score" inferior a 20 ##############')
print(df[df["Prestige_score"] < 20])

# %% [markdown]
# Podemos aplicar varias condiciones para que se vayan filtrando las filas y así quedarnos con las que nos interesan. Para ello podemos combinar los operadores *element-wise* para la conjunción (`&`), disyunción (`|`) y negación (`~`).
#
# Veamos un ejemplo donde se aplica la conjunción:

# %%
print(
    '\n############## Personas con "Prestige_score" inferior a 20 y escuela "vocational" ##############'
)
print(df[(df["Prestige_score"] < 20) & (df["Type_school"] == "vocational")])

# %% [markdown]
# Y ahora un último ejemplo con la negación:

# %%
print(
    '\n############## Personas con "Prestige_score" inferior a 20, escuela "vocational" y con certificado "no not_taken" ##############'
)
print(
    df[
        (df["Prestige_score"] < 20)
        & (df["Type_school"] == "vocational")
        & ~(df["Leaving_certificate"] == "not_taken")
    ]
)

# %% [markdown]
# ## 3.4 Concatenación de `DataFrames`
# Si queremos concatenar dos `DataFrames` podemos utilizar la función `concat()`.
#
# En el siguiente ejemplo se crean dos `DataFrames` pequeños, el primero con las filas 7 y 8 de `df` y el segundo con las filas 5, 6 y 7 para, posteriormente, concatenarlos.

# %%
print("\n############## se crea df1 con las filas 7 y 8 de df ##############")
df1 = df.loc[7:8]
print(df1)

print("\n############## se crea df2 con las filas 5, 6 y 7 de df ##############")
df2 = df.loc[5:7]
print(df2)

print("\n############## se concatenan df1 y df2 ##############")
print(pd.concat([df1, df2], axis=0))

# %% [markdown]
# Al poner el parámetro `axis` a 0 se concatenan verticalmente. Si lo hubiésemos puesto a 1 se concatenarían horizontalmente.
#
# Algo que nos puede llamar la atención es que aparecen los índices de las filas que originalmente tenían en `df` y, por tanto, en el ejemplo aparecen dos filas con el número 7. Cuando hagamos una catenación, si esos índices originales no son relevantes, lo mejor es utilizar el parámetro `ignore_index=True`, lo que hará que se reseteen los índices.

# %%
print(
    "\n############## se concatenan df1 y df2 y se resetean los índices ##############"
)
print(pd.concat([df1, df2], axis=0, ignore_index=True))


# %% [markdown]
# ## Ejercicios
#
# Haz un programa que cargue el fichero **biomed.data** (es un archivo de texto) y realice lo siguiente:
# 1. Agrupa por paciente ('Observation_number') y calcula los valores medios
# 2. Agrupa por paciente y clase y calcula la media de 'm1'
# 3. Agrupa por paciente y clase y calcula la media y desviación de la edad y el máximo y el mínimo de m4
# 4. Cambia la edad a meses y el nombre de la columna a 'Months' (busca ayuda de la función `rename()` para renombrar una columna)
# 5. Haz un filtro para ver las filas en las que 'm3' sea menor que 10 o mayor que 100 y que la clase NO sea normal
#
# Estos ejercicios no es necesario entregarlos.
