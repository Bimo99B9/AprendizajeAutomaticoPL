import pandas as pd
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

print("###################################################################")
print("1. Carga el fichero **heart_failure_clinical_records_dataset.csv** (es un archivo de texto). ")
print("###################################################################")
