import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

N_SPLITS = 3

data = pd.DataFrame()
for i in range(0, 9):
    split = pd.read_csv("../../../data/updrs_splits/split_{}.csv".format(i))
    data = pd.concat([data, split])

X = data.drop(columns=["patno", "score", "date_scan"])
y = data["score"].astype(int)

scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))

gb_classifier = GradientBoostingClassifier(n_estimators=2500, max_depth=4,
                                           learning_rate=0.1,
                                           max_features="log2")
scores = cross_val_score(gb_classifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(gb_classifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_micro')
print("F1-micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(gb_classifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_macro')
print("F1-macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
