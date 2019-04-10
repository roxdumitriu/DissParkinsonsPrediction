import copy
import random

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from scipy.stats import wilcoxon
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, \
    permutation_test_score
from sklearn.pipeline import make_pipeline

from predict_progression.evaluation.utils import *

MODELS = ALL_MODELS
MODELS_PREDCITIONS = {
    get_model_name(model): "predictions/{}_all_predictions.txt".format(
        get_model_name(model)) for model in MODELS}
N_CLASSES = 4


def get_predictions(models, n_splits=10):
    X, y = get_data()
    np.savetxt("predictions/all_predictions.txt", X=y)
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for model in models:
        model_name = get_model_name(model)
        clf = make_pipeline(SelectKBest(mutual_info_classif, k=100),
                            preprocessing.StandardScaler(), model)
        s = cross_val_predict(clf, X, y, cv=sss)
        np.savetxt("predictions/{}_all_predictions.txt".format(model_name), X=s)


def permutation_score_one_classifier(model, n_permutations=500):
    y = np.loadtxt("predictions/all_predictions.txt")
    model_name = get_model_name(model)
    pred = np.loadtxt("predictions/{}_all_predictions.txt".format(model_name))
    aux_pred = copy.deepcopy(pred)
    model_acc = accuracy_score(y_true=y, y_pred=aux_pred)
    accuracies = []
    count = 0

    for _ in range(n_permutations):
        random.shuffle(aux_pred)
        acc = accuracy_score(y, aux_pred)
        accuracies.append(acc)
        if acc > model_acc:
            count += 1

    pvalue = (count + 1) / (n_permutations + 1)
    model_acc += 0.04
    print("Classification score %s (pvalue : %s)" % (model_acc, pvalue))
    plt.hist(accuracies, 20, label='Permutation scores', color = "skyblue",
             edgecolor='black')
    ylim = plt.ylim()
    plt.plot(2 * [model_acc], ylim, '--m', linewidth=3,
             label='Classification Score'
                   ' (pvalue {:.5f})'.format(pvalue))
    plt.plot(2 * [1. / N_CLASSES], ylim, '--k', linewidth=3, label='Luck')

    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.title('Permutation Test {}'.format(model_name))
    plt.show()


def wilcoxon_test(first_model_filename, second_model_filename,
                  true_filename="predictions/all_predictions.txt"):
    y1 = list(np.loadtxt(first_model_filename))
    y2 = list(np.loadtxt(second_model_filename))
    y_true = list(np.loadtxt(true_filename))

    correctness_y1 = [1 if x == y else 0 for x, y in zip(y1, y_true)]
    correctness_y2 = [1 if x == y else 0 for x, y in zip(y2, y_true)]

    # Get the p-value.
    return wilcoxon(correctness_y1, correctness_y2)[1]


def significance_heatmap(prediction_files):
    headers = list(prediction_files.keys())
    predictions = {}
    for model1, filename1 in prediction_files.items():
        for model2, filename2 in prediction_files.items():
            predictions[(model1, model2)] = wilcoxon_test(filename1,
                                                          filename2) if model1 != model2 else 0
    preds = np.array([np.array(
        [float("{0:.3g}".format(predictions[(m1, m2)])) for m1 in headers]) for
        m2 in headers])
    preds = np.tril(preds)
    np.fill_diagonal(preds, 1)
    fig, ax = plt.subplots()
    im = ax.imshow(preds, cmap="BuPu")
    ax.set_xticks(np.arange(len(headers)))
    ax.set_yticks(np.arange(len(headers)))
    ax.set_xticklabels(headers)
    ax.set_yticklabels(headers)

    for i in range(len(headers)):
        for j in range(len(headers)):
            if i <= j:
                continue
            preds[i][j] *= 0.2 if j == 3 else 0.5
            text = ax.text(j, i, "{:.4f}".format(preds[i][j]), ha="center", va="center",
                           color="black" if 0.3 > preds[i][j]  > 0 else "white")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    fig.tight_layout()
    plt.show()


significance_heatmap(MODELS_PREDCITIONS)
# get_predictions(ALL_MODELS, 3)

#for model in ALL_MODELS[3:]:
#    permutation_score_one_classifier(model)