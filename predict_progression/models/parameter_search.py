import os

import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


def process_data(df):
    """ Process data to perform parameter search on.
     Parameters
     ----------
     df : Pandas DataFrame
         The data to be processed. Includes the labels.
     Returns
     ----------
     Features and labels, processed and scaled.
     """
    X = df.drop(columns=["patno", "score", "date_scan"])
    y = df["score"].astype(int)
    scaler = preprocessing.StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))
    return X, y


def parameter_search(model, parameters):
    """ Perform Grid Search for parameter search. Exhaustively searches the
        given parameter space and returns the best parameters.
     Parameters
     ----------
     model : Sklearn estimator
         The model to be tuned
     parameters : dict
         A dictionary of tuning parameters and possible values. For continuous
         parameters, some sensible discrete values should be passed.
     """
    train_df = pd.DataFrame()
    for x in range(0, 8):
        split = pd.read_csv("../../../data/updrs_splits/split_{}.csv".format(x))
        train_df = pd.concat([train_df, split])

    test_df = pd.read_csv("../../../data/updrs_splits/split_9.csv")

    X_train, y_train = process_data(train_df)
    X_test, y_test = process_data(test_df)
    score = "accuracy"
    clf = GridSearchCV(model, parameters, scoring=score,
                       return_train_score=True)

    print("# Tuning hyper-parameters for {}".format(score))
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set: \n {} \n".format(
        clf.best_params_))
    print("Grid scores on development set: \n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params, train_score in zip(means, stds,
                                              clf.cv_results_['params'],
                                              clf.cv_results_[
                                                  'mean_train_score']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("The score on the training set is {}".format(train_score))
