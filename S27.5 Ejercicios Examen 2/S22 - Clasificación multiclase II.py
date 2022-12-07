import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
    elif sistema == "SVC":
        sys = SVC()
    elif sistema == "LogisticRegression":
        sys = LogisticRegression()
    elif sistema == "ovo(rl)":
        sys = OneVsOneClassifier(LogisticRegression())
    elif sistema == "ovo(svc)":
        sys = OneVsOneClassifier(SVC())
    elif sistema == "ovr(rl)":
        sys = OneVsRestClassifier(LogisticRegression())
    elif sistema == "ovr(svc)":
        sys = OneVsRestClassifier(SVC())
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

    resultados = metrics.classification_report(y_test, y_pred, output_dict=True)

    return [
        resultados["accuracy"],
        resultados["weighted avg"]["precision"],
        resultados["weighted avg"]["recall"],
        resultados["weighted avg"]["f1-score"],
    ]


print("###################################################################")
print("1. Carga los datos de la pestaña FHR que hay en el fichero **CTG.xls**. ")
print("###################################################################")

df = pd.read_excel("S22 - Clasificación muticlase II\CTG.xls", sheet_name="FHR")
filas, columnas = df.shape

X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

class_names = ["A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP"]


print("###################################################################")
print(
    "2. Se debe calcular para los sistemas vistos en esta sesión (y en la sesión anterior) las métricas accuracy y las medias ponderadas de precision, recall y F1."
)
print("###################################################################")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y
)

sistemas = [
    "KNeighborsClassifier",
    "GaussianNB",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "SVC",
    "LogisticRegression",
    "ovo(rl)",
    "ovo(svc)",
    "ovr(rl)",
    "ovr(svc)",
]

resultados = np.empty((len(sistemas), 4))
i = 0
for sistema in sistemas:
    resultados[i, :] = entrena_y_evalua(X_train, X_test, y_train, y_test, sistema)
    i += 1

df_resultados = pd.DataFrame(
    resultados, index=sistemas, columns=["Accuracy", "Precision", "Recall", "F1-score"]
)
print(df_resultados)
