import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

N_SPLITS = 10
data = pd.read_csv("../../data//thickness_and_volume_data.csv")
le = preprocessing.LabelEncoder()

X = data.drop(columns=["patno", "date_scan", "diagnosis"])
y = data["diagnosis"]
le.fit(y)
# 0 = Healthy, 1 = PD
y = le.transform(y)
scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))

SVClassifier = svm.SVC(C=0.01, gamma=0.1, kernel='poly', degree=3, coef0=10.0)
scores = cross_val_score(SVClassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_micro')
print("F1-micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(SVClassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_macro')
print("F1-macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(SVClassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
