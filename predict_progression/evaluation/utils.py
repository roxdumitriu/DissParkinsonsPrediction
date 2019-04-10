import pandas as pd
from sklearn import preprocessing

import \
    predict_progression.models.GradientTreeBoosting.GradientBoostingClassifier as gtb
import predict_progression.models.KerasNN.KerasNN as keras
import predict_progression.models.MaxEnt.MaxEntClassifier as maxent
import predict_progression.models.RandomForest.RandomForestClassifier as rfc
import predict_progression.models.SGD.SGDclassifier as sgd
import predict_progression.models.SVM.SVM as svm

ALL_MODELS = [svm.get_model(), gtb.get_model(), rfc.get_model(), keras.get_model(),
              maxent.get_model(), sgd.get_model()]

DEFAULT_MODELS = [svm.get_default_model(), gtb.get_default_model(), rfc.get_default_model(), keras.get_default_model(),
              maxent.get_default_model(), sgd.get_default_model()]


def get_model_name(model):
    model_name = str(model).split("(")[0]
    if "KerasClassifier" in model_name:
        model_name = "MLP"
    return model_name


def get_data():
    data = pd.DataFrame()
    for i in range(0, 8):
        split = pd.read_csv("../../data/updrs_splits/split_{}.csv".format(i))
        data = pd.concat([data, split])

    X = data.drop(columns=["patno", "score", "date_scan"])
    y = data["score"].astype(int)

    return X, y


def train_and_test_from_splits(splits, test_index):
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    for i in range(0, len(splits)):
        if i == test_index:
            data_test = pd.read_csv(splits[i])
        else:
            data_train = pd.concat([data_train, pd.read_csv(splits[i])])
    X_train = data_train.drop(columns=["patno", "score", "date_scan"])
    X_test = data_test.drop(columns=["patno", "score", "date_scan"])

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test = pd.DataFrame(scaler.transform(X_test))

    y_train = data_train["score"].astype(int)
    y_test = data_test["score"].astype(int)
    return X_train, y_train, X_test, y_test
