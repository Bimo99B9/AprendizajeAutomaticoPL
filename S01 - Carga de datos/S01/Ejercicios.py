# 1. Cargando algún conjunto que ya venga en scikit-learn.

from sklearn import datasets
import pandas as pd

# Chemical analysis of wines grown in the same region in Italy.
cjto = datasets.load_wine()
print(cjto.DESCR)

X = cjto.data
Y = cjto.target

n_samples, n_features = X.shape
print("Number of samples: ", n_samples)
print("Number of features: ", n_features)

####
#2. Generando un conjunto artificial.

X, Y, coeficientes = datasets.make_regression(n_samples=100, n_features=5, 
                                        n_informative=3, n_targets=1, coef=True)

print("Coeficientes:\n", coeficientes)
print("Matriz de datos:\n", X[:3])
print("Clase:\n", Y[:4])


####
#3. Cargando desde un fichero de texto.

df = pd.read_csv('zoo.data', sep=',')
print(df)

####
#4. Cargando datos desde un excel.

df = pd.read_excel('Test.xlsx', header=0)
print(df)