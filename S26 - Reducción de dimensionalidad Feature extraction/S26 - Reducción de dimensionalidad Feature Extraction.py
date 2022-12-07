import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

print("###################################################################")
print(
    "1. Carga el conjunto de datos  **gastroenterology.data**"
)
print("###################################################################")

df = pd.read_csv(
    "S26 - Reducción de dimensionalidad Feature extraction\gastroenterology.data")
df = df.transpose()
# Después de transponer la clase está en la columna 1.
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print("###################################################################")
print(
    "2. Sepáralo en 80% para entrenar y 20% para test."
)
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234, stratify=y)

print("###################################################################")
print(
    "3. Prueba los algoritmos PCA y LinearDiscriminantAnalysis"
)
print("###################################################################")

print("Rendimiento inicial")
sys_tree = DecisionTreeClassifier(random_state=1234)
sys_tree.fit(X_train, y_train)

y_pred = sys_tree.predict(X_test)
acc_tree = metrics.accuracy_score(y_test, y_pred)
print("Accuracy árbol: %.4f" % acc_tree)

print("###################################################")
print("PCA")
num_componentes = 3
pca = PCA(n_components=num_componentes)
pca.fit(X_train)  # OJO, no necesita la clase para entrenar
X_train_pca = pca.transform(X_train)

print("Visualizar los datos")
if num_componentes == 2:
    plt.figure()
    [max_cp1, max_cp2] = X_train_pca.max(axis=0)
    [min_cp1, min_cp2] = X_train_pca.min(axis=0)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                c=y_train, cmap=plt.cm.Set1)
    plt.xlim([min_cp1, max_cp1 + 100])
    plt.ylim([min_cp2, max_cp2 + 100])
    plt.savefig("S26 - Reducción de dimensionalidad Feature extraction\src_pca_" +
                str(num_componentes)+'D.png')
elif num_componentes == 3:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_train_pca[:, 0], X_train_pca[:, 1],
                 X_train_pca[:, 2], c=y_train, cmap=plt.cm.Set1)
    plt.savefig("S26 - Reducción de dimensionalidad Feature extraction\src_pca_" +
                str(num_componentes)+'D.png')

print("Rendimiento con PCA - ", num_componentes, 'componentes')

X_test_pca = pca.transform(X_test)
sys_tree = DecisionTreeClassifier(random_state=1234)
sys_tree.fit(X_train_pca, y_train)
y_pred = sys_tree.predict(X_test_pca)
acc_tree = metrics.accuracy_score(y_test, y_pred)
print("Accuracy árbol: %.4f" % acc_tree)

print("###################################################")
print("LDA")

num_componentes = 2
lda = LinearDiscriminantAnalysis(n_components=num_componentes)
lda.fit(X_train, y_train)  # Necesita la clase.
X_train_lda = lda.transform(X_train)

print("Visualizar los datos")
if num_componentes == 2:
    plt.figure()
    [max_cp1, max_cp2] = X_train_lda.max(axis=0)
    [min_cp1, min_cp2] = X_train_lda.min(axis=0)
    plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1],
                c=y_train, cmap=plt.cm.Set1)
    plt.xlim([min_cp1, max_cp1])
    plt.ylim([min_cp2, max_cp2])
    plt.savefig("S26 - Reducción de dimensionalidad Feature extraction\src_lda_" +
                str(num_componentes)+'D.png')

print("Rendimiento con LDA - 2 componentes")

X_test_lda = lda.transform(X_test)
sys_tree = DecisionTreeClassifier(random_state=1234)
sys_tree.fit(X_train_lda, y_train)
y_pred = sys_tree.predict(X_test_lda)
acc_tree = metrics.accuracy_score(y_test, y_pred)
print("Accuracy árbol: %.4f" % acc_tree)

print("Visualizar los datos")
if num_componentes == 2:
    plt.figure()
    [max_cp1, max_cp2] = X_train_lda.max(axis=0)
    [min_cp1, min_cp2] = X_train_lda.min(axis=0)
    plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1],
                c=y_test, cmap=plt.cm.Set1)
    plt.xlim([min_cp1, max_cp1])
    plt.ylim([min_cp2, max_cp2])
    plt.savefig("S26 - Reducción de dimensionalidad Feature extraction\src_lda_" +
                str(num_componentes)+'D_test.png')
