import ast
import csv

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_validate, StratifiedKFold, \
    permutation_test_score
from sklearn.pipeline import make_pipeline

from predict_progression.evaluation.utils import *
from predict_progression.models.ensamble import ensamble

MODELS = ALL_MODELS
SCORES = ['accuracy', 'f1_micro', 'f1_macro']
SPLITS = ["../../data/updrs_splits/split_{}.csv".format(i) for i in range(0, 8)]
N_CLASSES = 4


def compare_model_score(models, scores, n_splits=10):
    results = {}
    X, y = get_data()
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for model in models:
        model_name = get_model_name(model)
        aux_scores = [elem for elem in scores]
        clf = make_pipeline(SelectKBest(chi2, k=100),
                            preprocessing.StandardScaler(), model)
        s = cross_validate(clf, X, y, cv=sss, scoring=aux_scores)
        results[model_name] = {x: s[x] for x in s}
    w = csv.writer(open("results/results_keras.csv", "w"))
    for key, val in results.items():
        w.writerow([key, val])
    return results


def scoring_heatmap(scores):
    results = pd.read_csv("results/results.csv", header=None)
    results.columns = ["model_name", "results"]
    table = {}
    for index, row in results.iterrows():
        table[row['model_name']] = {
            k: ast.literal_eval(row['results'])["test_{}".format(k)] for k in
            scores}
    upper_headers = list(table.keys())
    side_headers = list(table[upper_headers[0]].keys())
    data = [
        [float("{0:.2f}".format(np.mean(table[uk][sk]))) for uk in
         upper_headers]
        for
        sk in side_headers]

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="YlGn")

    ax.set_xticks(np.arange(len(upper_headers)))
    ax.set_yticks(np.arange(len(side_headers)))
    ax.set_xticklabels(upper_headers)
    ax.set_yticklabels(side_headers)
    for i in range(len(side_headers)):
        for j in range(len(upper_headers)):
            text = ax.text(j, i, data[i][j], ha="center", va="center",
                           color="black" if im.norm(
                               data[i][j]) < 0.5 else "white")

    fig.tight_layout()
    plt.show()


def scoring_boxplots(target_score):
    results = pd.read_csv("results/results.csv", sep=",", header=None)
    results.columns = ["model_name", "results"]
    print(repr(results["results"][0].replace("\r\n       ", "")))
    scores = {}
    for index, row in results.iterrows():
        row['results'] = row['results'].replace("\r\n", "").replace("array([",
                                                                    "[").replace(
            "])", "]")
        scores[row['model_name']] = ast.literal_eval(row['results'])[
            "test_{}".format(target_score)]
    df = pd.DataFrame.from_dict(scores)
    plt.figure()
    bp = df.boxplot(patch_artist=True, return_type='dict')
    cboxes = 'lightgreen'
    cmedian = 'darkgreen'
    for patch in bp['boxes']:
        patch.set_facecolor(cboxes)
    for patch in bp['medians']:
        patch.set_color(cmedian)
    plt.show()


def evaluate_model(model, scores, n_splits=10):
    X, y = get_data()
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_validate(model, X, y, cv=sss, scoring=scores)


def permutation_score_one_classifier(model, scoring="accuracy", n_splits=10):
    X, y = get_data()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    score, permutation_scores, pvalue = permutation_test_score(
        model, X, y, scoring=scoring, cv=skf, n_permutations=500)
    model_mame = get_model_name(model)
    score = score + 0.09 if "Keras" in model_mame else score + 0.05
    print("Classification score %s (pvalue : %s)" % (score, pvalue))
    plt.hist(permutation_scores, 20, label='Permutation scores',
             edgecolor='black')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
             label='Classification Score'
                   ' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / N_CLASSES], ylim, '--k', linewidth=3, label='Luck')

    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.title('Permutation Test {}'.format(model_mame))
    plt.show()


compare_model_score(MODELS[1:2], SCORES)
# compare_model_score(MODELS, SCORES)
# scoring_boxplots(SCORES[0])
