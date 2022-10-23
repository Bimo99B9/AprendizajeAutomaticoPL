print("###################################################################")
print("1. Carga el fichero **iris.data** (es un archivo de texto).")
print("###################################################################")

import pandas as pd
import matplotlib.pyplot as plt

cabecera = ["sepal length", "sepal width", "petal length", "petal width", "class"]
df = pd.read_csv("S04 - Gráficos/iris.data", names=cabecera, sep=",")
filas, columnas = df.shape

X = df.drop(["class"], axis=1)
y = df.pop("class")


print(
    "#############################################################################################"
)
print(
    "3. Haz una figura y representa la distribución de cada atributo con un histograma para cada uno"
)
print(
    "#############################################################################################"
)

num_atributos = df.shape[1]

fig, axs = plt.subplots(1, num_atributos)
fig.set_size_inches((num_atributos * 3, 3))

for i in range(num_atributos):
    axs[i].hist(X[X.columns[i]], 10)
    axs[i].set_title(X.columns[i])

plt.savefig("src_distribuciones.png")

print("###################################################################")
print(
    "4. Haz una figura en la que se enfrenten todas las variables dos-a-dos entre ellas con gráficos de dispersión"
)
print("###################################################################")

num_atributos = X.shape[1]

# Matriz de correlacion, n atributos X n atributos.
fig, axs = plt.subplots(num_atributos, num_atributos)
fig.set_size_inches(num_atributos * 3, num_atributos * 3)

ejem_clase_set = y == "Iris-setosa"
ejem_clase_ver = y == "Iris-versicolor"
ejem_clase_vir = y == "Iris-virginica"

for i in range(num_atributos):
    for j in range(num_atributos):
        axs[i][j].scatter(
            x=X[ejem_clase_set][X.columns[j]],
            y=X[ejem_clase_set][X.columns[i]],
            color="b",
            label="Iris-setosa",
        )
        axs[i][j].scatter(
            x=X[ejem_clase_ver][X.columns[j]],
            y=X[ejem_clase_ver][X.columns[i]],
            color="r",
            label="Iris-versicolor",
        )
        axs[i][j].scatter(
            x=X[ejem_clase_vir][X.columns[j]],
            y=X[ejem_clase_vir][X.columns[i]],
            color="g",
            label="Iris-virginica",
        )

        if j == 0:
            axs[i][j].set_ylabel(X.columns[i])
        if i == num_atributos - 1:
            axs[i][j].set_xlabel(X.columns[j])

        if i == 0 and j == 0:
            axs[i][j].legend(loc="best")

        axs[i][j].grid(axis="both", color="gray", linestyle="--")

plt.show()
