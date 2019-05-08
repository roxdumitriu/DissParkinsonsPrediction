import copy
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score

import predict_progression.evaluation.utils as utils
from predict_progression.evaluation import cross_validation


def permutation_test(model, n_permutations=500):
    """ Run the permutation test on the predictions of a model and plot the
        results as a histogram. The histogram shows the accuracy of a random
        permutation of the model's predictions.
     Parameters
     ----------
     model : Sklearn estimator
        The model to compute the permutation test on.
     n_permutations : int
        The number of permutations that should be performed.
     """
    y = cross_validation.get_true_labels()
    pred = cross_validation.get_model_predicted_labels(model)
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
    plt.hist(accuracies, 20, label='Permutation scores', color="skyblue",
             edgecolor='black')
    ylim = plt.ylim()
    plt.plot(2 * [model_acc], ylim, '--m', linewidth=3,
             label='Classification Score'
                   ' (pvalue {:.5f})'.format(pvalue))
    plt.plot(2 * [1. / 4], ylim, '--k', linewidth=3, label='Luck')

    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.title('Permutation Test {}'.format(utils.get_model_name(model)))
    plt.show()


def permutation_test_all_models(models=utils.ALL_MODELS, n_permutations=500):
    """ Run the permutation test on the predictions of a model and plot the
            results as a histogram. The histogram shows the accuracy of a random
            permutation of the model's predictions.
    Parameters
    ----------
    models : List of Sklearn estimators
        The list of models to compute permutation test on.
    n_permutations : int
        The number of permutations that should be performed.
    """
    for model in models:
        permutation_test(model, n_permutations)


def wilcoxon_test(first_model, second_model):
    """ Perform the pair-wise Wilcoxon test for significance testing of two
        models.
     Parameters
     ----------
     first_model : Sklearn estimator
        First model to perform the test on.
     second_model : Sklearn estimator
         Second model to perform the test on.
     """
    y1 = cross_validation.get_model_predicted_labels(first_model)
    y2 = cross_validation.get_model_predicted_labels(second_model)
    y_true = cross_validation.get_true_labels()

    correctness_y1 = [1 if x == y else 0 for x, y in zip(y1, y_true)]
    correctness_y2 = [1 if x == y else 0 for x, y in zip(y2, y_true)]

    # Get the p-value.
    return wilcoxon(correctness_y1, correctness_y2)[1]


def wilcoxon_heatmap(models=utils.ALL_MODELS):
    """ Plot the results of the Wilcoxon test as a heat map.
    """
    headers = [utils.get_model_name(m) for m in models]
    predictions = {}
    for model1, name1 in zip(models, headers):
        for model2, name2 in zip(models, headers):
            predictions[(name1, name2)] = wilcoxon_test(model1,
                                                        model2) if name1 != name2 else 0
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
            text = ax.text(j, i, "{:.4f}".format(preds[i][j]), ha="center",
                           va="center",
                           color="black" if 0.3 > preds[i][j] > 0 else "white")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    fig.tight_layout()
    plt.show()

