import numpy as np
import shap
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from predict_progression.evaluation import utils

MODELS = utils.ALL_MODELS
N_SPLITS = 10
EXPLAINER_PATH = utils.SHAP_EXPLAINER_PATH
SHAP_VALUES_PATH = utils.SHAP_VALUES_PATH


def save_explainer_results(X_train, Y_train, X_test,
                           explainer_path=EXPLAINER_PATH,
                           shap_values_path=SHAP_VALUES_PATH,
                           models=MODELS):
    """ Compute the shap values for one train-test split in n-fold cross
        validation, on all models. Save the explainer and shap values in an
        external file.
     Parameters
     ----------
     X_train : (n_samples, n_features) array
        The training data.
     Y_train : (n_samples, ) array
        The labels of the training data.
     X_test : (n_samples, n_features) array
        The test data.
     explainer_path : string
        Path to where to save the explainers of the models.
     shap_values_path : string
        Path to where to save the shap values of the models.
     models : list
        List of Sklearn estimators.
     """
    for model in models:
        model_name = utils.get_model_name(model)
        model = model.fit(X_train, Y_train)
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        shap_values = explainer.shap_values(X_test)
        np.savetxt(explainer_path.format(model_name), explainer.expected_value)
        for i in range(4):
            np.savetxt(shap_values_path.format(model_name, i), shap_values[i])


def save_explainer_results_per_fold(models=MODELS, n_splits=N_SPLITS):
    """ Compute the shap values for all n-fold cross validation splits, on all
        models.
     Parameters
     ----------
     models : list
        List of Sklearn estimators.
     n_splits : int
        Number of splits in cross validation.
     """
    X, y = utils.get_data()
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1,
                                 random_state=0)
    fold_num = 0
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_string = "fold_{}".format(fold_num)
        explainer_path = EXPLAINER_PATH + "_" + fold_string
        shap_values_path = SHAP_VALUES_PATH + "_" + fold_string
        save_explainer_results(X_train=X_train, X_test=X_test,
                               Y_train=y_train, models=models,
                               explainer_path=explainer_path,
                               shap_values_path=shap_values_path)


def average_shap_values(models=MODELS, n_splits=N_SPLITS):
    """ Average the shap values for all of the splits in n-fold cross
        validation. Stacks the cross validation results into one file and saves
        the result in one file.
     Parameters
     ----------
     models : list
        List of Sklearn estimators.
     n_splits : int
        Number of splits in cross validation.
     """
    for model in models:
        model_name = utils.get_model_name(model)
        sv = np.array([])
        for j in range(4):
            for i in range(n_splits):
                fold_string = "fold_{}".format(i)
                shap_values_path = SHAP_VALUES_PATH + "_" + fold_string
                s = np.loadtxt(shap_values_path.format(model_name, j))
                sv = s if len(sv) == 0 else np.vstack((sv, s))
            np.savetxt(SHAP_VALUES_PATH.format(model_name, j), sv)


def interpret_shap_values(models=MODELS):
    """ Plot the average of the shap values in a bar chart, ranking the features
        by their shap importance. Plots the top 20 features, with per-class
        importance.
     Parameters
     ----------
     models : list
        List of Sklearn estimators.
     Returns
     ----------
     A dict from model name to a dict of features and shap values.
     """
    X_train, X_test, Y_train, Y_test = utils.get_split_data()
    X_test = pd.DataFrame(np.around(X_test, decimals=3))
    for model in models:
        model_name = utils.get_model_name(model)
        print(model_name)
        shap_values = []
        for i in range(4):
            sv = np.loadtxt(SHAP_VALUES_PATH.format(model_name,i))
            shap_values.append(sv)
        shap.summary_plot(shap_values, X_test)


def average_shap_values_per_class(models=MODELS):
    """ Average total shap values for each model.
     Parameters
     ----------
     models : list
        List of Sklearn estimators.
     """
    models_average_shap_per_class = {}
    for model in models:
        feature_shap_values = {}
        model_name = utils.get_model_name(model)
        sv = []
        for i in range(4):
            class_name = "Class {}".format(i)
            sv_class = {}
            s = np.loadtxt(
                "results/interpretability/{}_shap_values_{}".format(model_name,
                                                                    i))
            sv = s if len(sv) == 0 else np.vstack((sv, s))
            for j in range(len(s[0])):
                feature_values = np.array([abs(s[k][j]) for k in range(len(s))])
                sv_class["Feature {}".format(j)] = (
                    np.mean(feature_values), np.std(feature_values))
            sorted_sv_class = sorted(sv_class.items(), key=lambda kv: -kv[1][0])
            feature_shap_values[class_name] = sorted_sv_class
        models_average_shap_per_class[model_name] = feature_shap_values
        print(model_name, feature_shap_values)
    return models_average_shap_per_class


# interpret_shap_values(models=MODELS)

