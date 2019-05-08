import ast
import csv

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, \
    cross_val_predict

import predict_progression.evaluation.utils as utils


def compute_evaluation_metrics(scores=utils.SCORES, models=utils.ALL_MODELS,
                               results_path=utils.RESULTS_FILE_PATH):
    """ Compute the 10-fold cross validation evaluation metrics of the models
        and save them in a CSV file for later reference.
     Parameters
     ----------
     scores : list
         The list of metrics to be computed. Defaults to accuracy, f1 micro and
         f1 macro.
     models : list
         The list of models to evaluate.
     results_path : string
         Gives the CSV file in which to save the results for quicker analysis.
     """
    results = {}
    X, y = utils.get_data()
    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for model in models:
        model_name = utils.get_model_name(model)
        print("Training {}".format(model_name))
        aux_scores = [elem for elem in scores]
        clf = utils.pipeline_model(model)
        s = cross_validate(clf, X, y, cv=sss, scoring=aux_scores, return_train_score=False)
        results[model_name] = {x: s[x] for x in s}
        print()
        print("Finished training {}".format(model_name))
    w = csv.writer(open(results_path, "w"))
    for key, val in results.items():
        w.writerow([key, val])
    return results


def get_evaluation_metric(target_score, results_path=utils.RESULTS_FILE_PATH):
    """ Read the evaluation metrics from the results CSV.
     Parameters
     ----------
     target_score : string
         Which score to return.
     results_path : string
         The path that store the evaluation results.
     Returns
     ----------
     A dict from model name to a list of scores, one for each fold.
     """
    results = pd.read_csv(results_path, sep=",", header=None)
    results.columns = ["model_name", "results"]
    scores = {}
    for index, row in results.iterrows():
        row['results'] = row['results'].replace("\r\n", "").replace("array([",
                                                                    "[").replace(
            "])", "]")
        scores[row['model_name']] = ast.literal_eval(row['results'])[
            "test_{}".format(target_score)]
    return scores


def compute_predictions_all_models(models, prediction_path=utils.PREDICTIONS_PATH):
    """ Compute the 10-fold cross validation predictions of the
        models and store them in a CSV. The predictions are computed after
        each training routine and concatenated into one.
     Parameters
     ----------
     models : list
         The list of models to evaluate.
     prediction_path : string
         The path that store the predictions.
     """
    X, y = utils.get_data()
    np.savetxt(prediction_path + "/all_predictions.txt", X=y)
    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for model in models:
        model_name = utils.get_model_name(model)
        clf = utils.pipeline_model(model)
        s = cross_val_predict(clf, X, y, cv=sss)
        np.savetxt(
            prediction_path + "/{}_all_predictions.txt".format(model_name), X=s)


def get_true_labels(prediction_path=utils.PREDICTIONS_PATH):
    """ Read and return the true labels for all data samples.
    """
    return np.loadtxt(prediction_path + "/all_predictions.txt")


def get_model_predicted_labels(model, prediction_path=utils.PREDICTIONS_PATH):
    """ Read and return the model predictions for all data samples.
    """
    model_name = utils.get_model_name(model)
    return np.loadtxt(
        prediction_path + "/{}_all_predictions.txt".format(model_name))
