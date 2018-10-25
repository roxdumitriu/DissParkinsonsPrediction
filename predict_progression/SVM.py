import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score

N_SPLITS = 1
data = pd.read_csv("../data/updrs.csv")
le = preprocessing.LabelEncoder()

X = data.drop(columns=["patno", "score", "date_scan"])
y = data["score"].astype(int)
SVClassifier = svm.SVC(kernel='rbf', gamma='auto')
scores = cross_val_score(SVClassifier, X, y, cv=10, scoring='f1_micro')
print("F1-micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(SVClassifier, X, y, cv=10, scoring='f1_macro')
print("F1-macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(SVClassifier, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
