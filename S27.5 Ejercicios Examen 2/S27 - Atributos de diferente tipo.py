import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC


print("###################################################################")
print("1. Carga el conjunto de datos  **CreditApproval.data**")
print("###################################################################")

cabecera = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
    "A09",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "class",
]

df = pd.read_csv(
    "S27 - Atributos de diferente tipo\CreditApproval.data",
    sep=",",
    names=cabecera,
    na_values="?",
)
filas, columnas = df.shape

class_enc = LabelEncoder()

df["class"] = class_enc.fit_transform(df["class"])

print("Clases: ", class_enc.classes_)
print(df.head(5))

print("###################################################################")
print(
    "2. Selecciona 5 ejemplos y vete realizando las codificaciones paso a paso como hemos visto en la práctica. Que no os asuste ver datos con caracteres, serán atributos categóricos o binarios y se tratan como ya hemos visto."
)
print("###################################################################")

df5 = df.sample(random_state=1, n=5)
X = df5.iloc[:, 0 : (columnas - 1)]
y = df5.iloc[:, (columnas - 1)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# A1:	b, a.
# A2:	continuous.
# A3:	continuous.
# A4:	u, y, l, t.
# A5:	g, p, gg.
# A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
# A7:	v, h, bb, j, n, z, dd, ff, o.
# A8:	continuous.
# A9:	t, f.
# A10:	t, f.
# A11:	continuous.
# A12:	t, f.
# A13:	g, p, s.
# A14:	continuous.
# A15:	continuous.
# class: 0,1         (class attribute)

# Identificar los grupos de atributos a preprocesar.
atr_bina = ["A01", "A09", "A10", "A12"]
atr_cate = ["A04", "A05", "A06", "A07", "A13"]
atr_nume = ["A02", "A03", "A08", "A11", "A14", "A15"]


print("###################################################################")
print(
    "3. Para los atributos binarios se puede utiliza OneHotEncoder() utilizando el parámetro `drop` convenientemente. La clase se codifica al principio, nada más cargar los datos"
)
print("###################################################################")

print("Atributos binarios")
pipe_bina = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("bin", OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first")),
    ]
)
# Entrenar sobre A01
pipe_bina.fit(X_train["A01"].values.reshape(-1, 1))
pipe_bina.transform(X_train["A01"].values.reshape(-1, 1))

print("Atributos categóricos")
pipe_cate = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

# Entrenar sobre A06
pipe_cate.fit(X_train["A06"].values.reshape(-1, 1))
pipe_cate.transform(X_train["A06"].values.reshape(-1, 1))

# Binarios y categóricos se podrían tratar de manera conjunta utilizando
# OneHotEncoder(handle_unknown="ignore", sparse=False, drop"if_binary")

print("Atributos numéricos")
pipe_nume = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("std", StandardScaler())]
)
pipe_nume.fit(X_train["A03"].values.reshape(-1, 1))
pipe_nume.transform(X_train["A03"].values.reshape(-1, 1))

print("Column Transformer")

# Definir un ColumnTransformer para tratar los atributos de manera diferente
at_transformer = ColumnTransformer(
    [
        ("at_bin", pipe_bina, atr_bina),
        ("at_cat", pipe_cate, atr_cate),
        ("at_num", pipe_nume, atr_nume),
    ],
    remainder="passthrough",
)

at_transformer.fit(X_train)

# Preparar nombres de columnas.
nombres_oh_atr_cat = at_transformer.named_transformers_["at_cat"][
    "oh"
].get_feature_names_out(atr_cate)
nombres_columnas = np.append(atr_bina, nombres_oh_atr_cat)
nombres_columnas = np.append(nombres_columnas, atr_nume)
print(nombres_columnas)

print("############# Conjunto de entrenamiento #############")
print(X_train)
print(pd.DataFrame(at_transformer.transform(X_train), columns=nombres_columnas))

print("############# Conjunto de test #############")
print(X_test)
print(pd.DataFrame(at_transformer.transform(X_test), columns=nombres_columnas))

print("############# Pipeline con clasificador #############")
# pipeline que realiza el preprocesado y lo encadena con un estimador.
sys_prep_svc = Pipeline(
    [
        ("atr_trans", at_transformer),
        ("sys", SVC()),
    ]
)

sys_prep_svc.fit(X_train, y_train)
y_train_pred = sys_prep_svc.predict(X_train)
print("Predicciones en el conjunto de entrenamiento: ", y_train_pred)
y_test_pred = sys_prep_svc.predict(X_test)
print("Predicciones en el conjunto de test:     ", y_test_pred)

print("###################################################################")
print(
    "4. Una vez que veas que la codificación funciona haz una búsqueda de hiperparámetros utilizando todos los ejemplos"
)
print("###################################################################")

# todos los datos.
X = df.iloc[:, 0 : (columnas - 1)]
y = df.iloc[:, (columnas - 1)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

print("############# Grid Search #############")

# Crear un generador de folds partiendo el conjunto en 5 trozos.
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

# Crear una GridSearch para el SVC donde le pasamos los hiperparámetros que queremos probar.
param = {"sys__C": [0.01, 0.1, 1, 10, 100], "sys__gamma": [0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(
    sys_prep_svc, param_grid=param, scoring="accuracy", cv=folds, verbose=1, n_jobs=-1
)

best_model = gs.fit(X_train, y_train)

print("Mejor combinación de hiperparámetros: ", best_model.best_params_)
print("Mejor rendimiento obtenido: %.4f" % best_model.best_score_)

y_pred = best_model.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
