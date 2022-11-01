import pandas as pd
from sklearn import impute, metrics, preprocessing
from sklearn.tree import DecisionTreeClassifier

"""
Aprendizaje Automático I - Primer Examen PL
Daniel López Gala - UO281798
"""

atributos = []
for i in range(24):
    atributos.append(f"atr{i}")
atributos.append("Y")
# print(len(atributos))

df = pd.read_csv("datos.data", na_values="?", sep=",", names=atributos)
filas, columnas = df.shape

# print(df.head())
# print(df.describe())

print("\n################# Tratar clase Y #################")
# Nota: No determino "Turn" o "Forwards" como clase positiva teniendo en cuenta cuál es la minoritaria porque no conozco cuál es la más importante.
# Si fuesen decisiones de un coche (Turn o forward en la dirección) ambas podrían ser igual de importantes, por lo que al no haber una claramente minoritaria,
# consideré que no era necesario indicarlo.

label_enc = preprocessing.LabelEncoder()
df["Y"] = label_enc.fit_transform(df["Y"])
print(df["Y"])

print("\n################# Tratar valores desconocidos #################")
# Máscara que identifica las filas que tienen al menos un True
mask = df.isnull().any(axis=1)
print("Ejemplos sin corregir.")
print(df[mask])
imputer_media = impute.KNNImputer()
df[df.columns] = imputer_media.fit_transform(df[df.columns])
print("Ejemplos corregidos.")
print(df[mask])
# Comprobación de que las clases están bien codificadas y no hay valores intermedios.
print(df["Y"].unique())


print("\n################# Árbol de decisión #################")
# Reescritura = Evaluar sobre el mismo conjunto de datos. (No hay que separar en train y test).
X = df.iloc[:, 0 : (columnas - 1)]
print(X)
y = df.iloc[:, (columnas - 1)]
print(y)

# Árbol limitando la profundidad a 3.
sys_dt_depth = DecisionTreeClassifier(random_state=4321, max_depth=3)
# se entrena el árbol de decisión
sys_dt_depth.fit(X, y)
y_pred = sys_dt_depth.predict(X)
print("Accuracy : %.4f" % metrics.accuracy_score(y, y_pred))


print("\n\n################# Preguntas ################# ")

print("\na. La accuracy obtenida, ¿es la esperada si ponemos el sistema en producción?")
"""
No, ya que hemos calculado su accuracy enfrentándose a casos ya vistos en el entrenamiento. Por tanto, nuestro árbol está preparado para esos casos, pero no sabemos cómo se comportaría ante casos no vistos. De hecho, si no limitasemos la profundidad del árbol, este acertaría al predecir cualquiera de los casos ya vistos en el entrenamiento, pues sus ramas se extenderían hasta estos.
"""
sys_dt = DecisionTreeClassifier(random_state=4321)
sys_dt.fit(X, y)
y_pred_dt = sys_dt.predict(X)
print("Accuracy : %.4f" % metrics.accuracy_score(y, y_pred_dt))
"""
Para conseguir una accuracy más realista deberíamos probar el árbol con casos no vistos, mediante validación cruzada, hold-out, o leave-one-out.
"""

print(
    "\nb. ¿Qué te parecen los números que aparecen en la matriz de confusión? ¿Qué crees que hace bien y/o mal el modelo entrenado?"
)
class_labels = df["Y"].unique()
print("Clases: ", class_labels)
cm = metrics.confusion_matrix(y, y_pred, labels=class_labels)
print("Matriz de confusión: ")
print(cm)

"""
Los falsos positivos son mayores que los falsos negativos.
"""
print("Precision: %0.2f" % metrics.precision_score(y, y_pred))
"""
Esto quiere decir que el 85 por ciento de los ejemplos a los que se les asigna la clase '1', son de la clase 1.
"""

print("Recall:", metrics.recall_score(y, y_pred))
"""
Y un 95 por ciento de los ejemplos de clase positiva son etiquetados correctamente.
"""

print(y.value_counts())
"""
Teniendo en cuenta que la clase 1 escogida es superior a la 0, tiene sentido que los falsos positivos sean mayores que los falsos negativos. Por tanto, el modelo etiqueta bien los casos positivos, y algo peor los negativos, siempre teniendo en cuenta que esto ocurre en casos ya vistos y desconocemos su funcionamiento en producción.
"""
