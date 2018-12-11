import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, \
    StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("../data/updrs.csv")
X = data.drop(columns=["patno", "score", "date_scan"])
y = data["score"].astype(int)
scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

parameters = [{'alpha': [0.1, 0.01, 0.001],
               'hidden_layer_sizes': [(64, 64, 64), (64, 64, 64, 64)],
               'max_iter': [5000, 10000]},
              ]
score = "accuracy"

clf = GridSearchCV(MLPClassifier(), parameters,
                   cv=StratifiedShuffleSplit(n_splits=10), scoring=score)

clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()
