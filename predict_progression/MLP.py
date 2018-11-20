import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

N_SPLITS = 10
data = pd.read_csv("../data/updrs.csv")
X = data.drop(columns=["patno", "score", "date_scan"])
y = data["score"].astype(int)

scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))
skf = StratifiedShuffleSplit(n_splits=N_SPLITS)

MLPclassifier = MLPClassifier(hidden_layer_sizes=(64, 64, 64, 64), alpha=0.1, max_iter=10000)
scores = cross_val_score(MLPclassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_micro')
print("F1-micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(MLPclassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_macro')
print("F1-macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(MLPclassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
