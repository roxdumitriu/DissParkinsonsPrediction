import os

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import \
    predict_progression.models.GradientTreeBoosting.GradientTreeBoosting as gtb
import predict_progression.models.KerasNN.KerasNN as keras
import predict_progression.models.MaxEnt.MaxEntClassifier as maxent
import predict_progression.models.RandomForest.RandomForestClassifier as rfc
import predict_progression.models.SGD.SGDclassifier as sgd
import predict_progression.models.SVM.SVM as svm

ALL_MODELS = [svm.get_model(), gtb.get_model(), rfc.get_model(),
              keras.get_model(),
              maxent.get_model(), sgd.get_model()]

DEFAULT_MODELS = [svm.get_default_model(), gtb.get_default_model(),
                  rfc.get_default_model(), keras.get_default_model(),
                  maxent.get_default_model(), sgd.get_default_model()]

SCORES = ["accuracy", "f1_micro", "f1_macro"]


def get_evaluation_absolute_path():
    script_path = os.path.abspath(__file__)
    # Remove the script name from the path.
    script_dir = os.path.split(script_path)[0]
    return script_dir


EVAL_PATH = get_evaluation_absolute_path()

RESULTS_FILE_PATH = os.path.join(EVAL_PATH, "results/results.csv")
DEFAULT_RESULTS_FILE_PATH = os.path.join(EVAL_PATH,
                                         "results/results_default.csv")
ENSEMBLE_RESULTS_FILE_PATH = os.path.join(EVAL_PATH,
                                          "results/results_ensemble.csv")
PREDICTIONS_PATH = os.path.join(EVAL_PATH, "predictions")
DATA_PATH = os.path.join(EVAL_PATH, "../../data")
SHAP_EXPLAINER_PATH = os.path.join(EVAL_PATH,
                                   "results/interpretability/{}_explainer")
SHAP_VALUES_PATH = os.path.join(EVAL_PATH,
                                "results/interpretability/{}_shap_values_{}")


def get_model_name(model):
    """ Give a string name to each estimator object.
    """
    model_name = str(model).split("(")[0]
    # If the estimator is a Keras wrapper, give it a more suggestive name, i.e.
    # MLP. Otherwise, infer the name from the object type.
    if "KerasClassifier" in model_name or "keras" in model_name:
        model_name = "MLP"
    return model_name


ALL_MODELS_DICT = {get_model_name(i): i for i in ALL_MODELS}
DEFAULT_MODELS_DICT = {get_model_name(i): i for i in ALL_MODELS}


def pipeline_model(model):
    """ Make a pipeline that performs feature selection and scales the data.
    """
    return make_pipeline(SelectKBest(mutual_info_classif, k=100),
                         preprocessing.StandardScaler(), model)


def get_data():
    """ Return the UPDRS data as one data frame. Excludes the validation corpus.
    """
    data = pd.DataFrame()
    for i in range(0, 8):
        split_path = os.path.join(DATA_PATH,
                                  "updrs_splits/split_{}.csv".format(i))
        # split = pd.read_csv("../../data/updrs_splits/split_{}.csv".format(i))
        split = pd.read_csv(split_path)
        data = pd.concat([data, split])

    X = data.drop(columns=["patno", "score", "date_scan"])
    y = data["score"].astype(int)

    return X, y


def get_split_data():
    """ Get a split of train and test data. The test data is 10% and train data
        is 90%. Data is already scaled using the standard scaler.
     """
    X, y = get_data()
    X_cols = list(X.columns.values)

    selector = SelectKBest(mutual_info_classif, k=100)
    X = selector.fit_transform(X, y)
    mask = selector.get_support(indices=True)
    columns = [X_cols[i] for i in range(len(X_cols)) if i in mask]
    X = pd.DataFrame(X)
    X.columns = columns

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1,
                                                        stratify=y,
                                                        random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test
