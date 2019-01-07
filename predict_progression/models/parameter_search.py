import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


def process_data(df):
    X = df.drop(columns=["patno", "score", "date_scan"])
    y = df["score"].astype(int)
    scaler = preprocessing.StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))

    return X, y


def parameter_search(model, parameters, n_splits):
    train_df = pd.DataFrame()
    for x in range(0, 9):
        split = pd.read_csv("../../../data/updrs_splits/split_{}.csv".format(x))
        train_df = pd.concat([train_df, split])

    test_df = pd.read_csv("../../../data/updrs_splits/split_9.csv")

    X_train, y_train = process_data(train_df)
    X_test, y_test = process_data(test_df)
    score = "accuracy"

    clf = GridSearchCV(model, parameters,
                       cv=StratifiedShuffleSplit(n_splits=n_splits),
                       scoring=score)

    print("# Tuning hyper-parameters for %s \n" % score)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set: \n {} \n".format(
        clf.best_params_))
    print("Grid scores on development set: \n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("Confusion matrix: \n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(confusion_matrix(y_true, y_pred))
    print()
