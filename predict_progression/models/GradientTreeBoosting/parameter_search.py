import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier


def process_data(df):
    X = df.drop(columns=["patno", "score", "date_scan"])
    y = df["score"].astype(int)
    scaler = preprocessing.StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))

    return X, y


train_df = pd.DataFrame()
for x in range(0, 9):
    split = pd.read_csv("../../../data/updrs_splits/split_{}.csv".format(x))
    train_df = pd.concat([train_df, split])

test_df = pd.read_csv("../../../data/updrs_splits/split_9.csv")

X_train, y_train = process_data(train_df)
X_test, y_test = process_data(test_df)

parameters = [{'n_estimators': [10, 50, 100, 1000, 2000, 3000],
               'learning_rate': [0.1, 0.01, 0.001],
               'max_depth': [2, 4, 8, 16],
               'max_features': [None, "auto", "log2"]}
              ]
score = "accuracy"

clf = GridSearchCV(GradientBoostingClassifier(), parameters,
                   cv=StratifiedShuffleSplit(n_splits=10), scoring=score)

print("# Tuning hyper-parameters for %s" % score)
print()

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
print(confusion_matrix(y_true, y_pred))
print()