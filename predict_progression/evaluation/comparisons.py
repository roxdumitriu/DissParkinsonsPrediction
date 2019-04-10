import ast
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline

from predict_progression.evaluation.utils import *
import predict_progression.models.ensemble.ensemble as ensemble

MODELS = ALL_MODELS
DEFAULT_MODELS = DEFAULT_MODELS
SCORES = ['accuracy', 'f1_micro', 'f1_macro']
SPLITS = ["../../data/updrs_splits/split_{}.csv".format(i) for i in range(0, 8)]
N_CLASSES = 4


def compare_model_score(models, scores, n_splits=10,
                        results_path="results/results.csv"):
    results = {}
    X, y = get_data()
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for model in models:
        model_name = get_model_name(model)
        aux_scores = [elem for elem in scores]
        clf = make_pipeline(SelectKBest(mutual_info_classif, k=100),
                            preprocessing.StandardScaler(), model)
        s = cross_validate(clf, X, y, cv=sss, scoring=aux_scores)
        results[model_name] = {x: s[x] for x in s}
    w = csv.writer(open(results_path, "w"))
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


def create_boxplts(df, y_label):
    plt.figure()
    bp = df.boxplot(patch_artist=True, return_type='dict',
                    medianprops={'linewidth': 2})
    edge_color = 'purple'
    fill_color = 'white'
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    plt.setp(bp['medians'], color='orange')
    plt.setp(bp['caps'], color='blue')
    plt.ylabel(y_label)
    plt.legend(
        [bp['caps'][0], bp['boxes'][0], bp['medians'][0], bp['fliers'][0]],
        ['min-max', '25% - 75%', 'median', 'outliers'])
    plt.show()


def scoring_boxplots(target_score, results_path="results/results.csv"):
    results = pd.read_csv(results_path, sep=",", header=None)
    results.columns = ["model_name", "results"]
    scores = {}
    for index, row in results.iterrows():
        row['results'] = row['results'].replace("\r\n", "").replace("array([",
                                                                    "[").replace(
            "])", "]")
        scores[row['model_name']] = ast.literal_eval(row['results'])[
            "test_{}".format(target_score)]
    df = pd.DataFrame.from_dict(scores)
    create_boxplts(df, target_score)


def hyperparam_tuning_improvement_boxplots(target_score, results_path,
                                           default_results_path):
    results = pd.read_csv(results_path, sep=",", header=None)
    results.columns = ["model_name", "results"]
    default_results = pd.read_csv(default_results_path, sep=",", header=None)
    default_results.columns = ["model_name", "results"]
    scores = {}
    for index, row in results.iterrows():
        row['results'] = row['results'].replace("\r\n", "").replace("array([",
                                                                    "[").replace(
            "])", "]")
        scores[row['model_name']] = ast.literal_eval(row['results'])[
            "test_{}".format(target_score)]
    default_scores = {}
    for index, row in default_results.iterrows():
        row['results'] = row['results'].replace("\r\n", "").replace("array([",
                                                                    "[").replace(
            "])", "]")
        default_scores[row['model_name']] = ast.literal_eval(row['results'])[
            "test_{}".format(target_score)]

    scores = {key: [abs(scores[key][i] - default_scores[key][i]) for i in
                    range(len(scores[key]))] for key in scores.keys()}

    df = pd.DataFrame.from_dict(scores)
    create_boxplts(df, "{} improvement".format(target_score))


def bar_chart(data, colours):
    ax = plt.subplot(111)
    increment = 0
    for (model_name, results), colour in zip(data.items(), colours):
        range_y = range(len(results))
        x = [x + increment for x in range(len(results))]
        bars = ["Fold {}".format(x) for x in range_y]
        ax.bar(x=x, height=results, width=0.2, align='center', label=model_name,
               color=colour, edgecolor='k')
        plt.xticks(range(len(results)), bars, color='k', rotation=45,
                   horizontalalignment='right')
        increment += 0.2
    plt.legend()

    plt.show()


def per_fold_bar_chart(models, target_score='accuracy'):
    results_single = pd.read_csv("results/results.csv", sep=",", header=None)
    results_ensemble = pd.read_csv("results/results_ensemble.csv", sep=",",
                                   header=None)

    results = pd.concat([results_single, results_ensemble])
    results.columns = ["model_name", "results"]

    scores = {}
    for index, row in results.iterrows():
        row['results'] = row['results'].replace("\r\n", "").replace("array([",
                                                                    "[").replace(
            "])", "]")
        scores[row['model_name']] = ast.literal_eval(row['results'])[
            "test_{}".format(target_score)]

    colours = ["c", "purple", "darkorange", "pink"]
    bar_chart({k: scores[k] for k in
               models}, colours)


def performance_heatmap(target_score="accuracy"):
    results = pd.read_csv("results/results.csv", header=None)
    results.columns = ["model_name", "results"]
    scores = {}
    for index, row in results.iterrows():
        row['results'] = row['results'].replace("\r\n", "").replace("array([",
                                                                    "[").replace(
            "])", "]")
        scores[row['model_name']] = ast.literal_eval(row['results'])[
            "test_{}".format(target_score)]
    upper_headers = list(scores.keys())
    side_headers = list(scores.keys())

    data = {uk: {sk: len(
        [i for i in range(len(scores[uk])) if
         scores[uk][i] > scores[sk][i] + 0.01])
        for sk in side_headers} for uk in upper_headers}

    clfs = sorted(upper_headers,
                  key=lambda x: sum([data[x][sk] for sk in side_headers]))
    data = [[data[clfu][clfside] * 10 for clfside in clfs] for clfu in clfs]

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="BuPu")

    ax.set_xticks(np.arange(len(clfs)))
    ax.set_yticks(np.arange(len(clfs)))
    ax.set_xticklabels(clfs)
    ax.set_yticklabels(clfs)
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            if i == j:
                continue
            text = ax.text(j, i, "{}%".format(data[i][j]), ha="center",
                           va="center",
                           color="black" if im.norm(
                               data[i][j]) < 0.5 else "white")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.xlabel("Losses")
    plt.ylabel("Wins")
    fig.tight_layout()
    plt.show()


def evaluate_model(model, scores, n_splits=10):
    X, y = get_data()
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = make_pipeline(SelectKBest(mutual_info_classif, k=100),
                        preprocessing.StandardScaler(), model)
    return cross_validate(clf, X, y, cv=sss, scoring=scores)


# performance_heatmap()
# compare_model_score(DEFAULT_MODELS, scores=SCORES,
#                     results_path="results/results_default.csv")

hyperparam_tuning_improvement_boxplots("accuracy", "results/results.csv",
                                       "results/results_default.csv")
