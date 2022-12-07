import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def entrena_y_evalua(X_train, X_test, y_train, y_test, sistema):
    if sistema == "KNeighborsClassifier":
        sys = KNeighborsClassifier()
    elif sistema == "GaussianNB":
        sys = GaussianNB()
    elif sistema == "DecisionTreeClassifier":
        sys = DecisionTreeClassifier(random_state=1234)
    elif sistema == "RandomForestClassifier":
        sys = RandomForestClassifier(random_state=1234, n_jobs=-1)
    else:
        print("Sistema no reconocido")
        exit()

    print("\n################################################")
    print("### %s" % sys)
    print("#################################################")

    model = sys.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    resultados = metrics.classification_report(
        y_test, y_pred, output_dict=True)

    return [
        resultados["accuracy"],
        resultados["weighted avg"]["precision"],
        resultados["weighted avg"]["recall"],
        resultados["weighted avg"]["f1-score"],
    ]


print("###################################################################")
print("1. Cargar conjunto de datos.")
print("###################################################################")

df = pd.read_excel(
    "S21 - Clasificación multiclase I\CTG.xls", sheet_name="NSP")
filas, columnas = df.shape

X = df.iloc[:, 0: (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

class_names = ["Normal", "Suspect", "Pathologic"]


print("###################################################################")
print("2. Hold-out 70-30")
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y
)

sistemas = [
    "KNeighborsClassifier",
    "GaussianNB",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
]

resultados = np.empty((len(sistemas), 4))
i = 0
for sistema in sistemas:
    resultados[i, :] = entrena_y_evalua(
        X_train, X_test, y_train, y_test, sistema)
    i += 1

df_resultados = pd.DataFrame(
    resultados, index=sistemas, columns=[
        "Accuracy", "Precision", "Recall", "F1-score"]
)
print(df_resultados)
