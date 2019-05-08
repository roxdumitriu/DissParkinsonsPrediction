import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from predict_progression.evaluation import utils, cross_validation
from predict_progression.evaluation.utils import ALL_MODELS_DICT

ENSEMBLE_MODELS = [ALL_MODELS_DICT["GradientBoostingClassifier"],
                   ALL_MODELS_DICT["MLP"], ALL_MODELS_DICT["SGDClassifier"]]


def _draw_boxplots(df, y_label):
    """ Draw boxplots from a Pandas data-frame. Used to unify the box plot style.
    """
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


def _draw_bar_chart(data, colours):
    """ Draw a bar chart from a Pandas data-frame or dict. Used to unify the bar
        chart style. One colour for each key in the data must be provided.
    """
    ax = plt.subplot(111)
    increment = 0
    for (model_name, results), colour in zip(data.items(), colours):
        range_y = range(len(results))
        x = [x + increment for x in range(len(results))]
        bars = ["Fold {}".format(x) for x in range_y]
        ax.bar(x=x, height=results, width=0.2, align='center', label=model_name,
               color=colour, edgecolor="k")
        plt.xticks(range(len(results)), bars, color='k', rotation=45,
                   horizontalalignment='right')
        increment += 0.2
    plt.legend()

    plt.show()


def scoring_boxplots(target_score="accuracy",
                     results_path=utils.RESULTS_FILE_PATH):
    """ Plot comparative box plots for all the classifiers, based on their 10-
        fold cross validation metric. Box plots show mean, standard deviation,
        min-max and outliers of the target score on the 10 folds.
     Parameters
     ----------
     target_score : string
         The score to base the box plots on.
     results_path : string
         The file from which the results can be retrieved.
     """
    scores = cross_validation.get_evaluation_metric(target_score, results_path)
    df = pd.DataFrame.from_dict(scores)
    _draw_boxplots(df, target_score)


def hyperparam_tuning_improvement_boxplots(target_score="Accuracy",
                                           results_path=utils.RESULTS_FILE_PATH,
                                           default_results_path=
                                           utils.DEFAULT_RESULTS_FILE_PATH):
    """ Plot comparative box plots for all the classifiers, based on their
        improvement in performance after parameter tuning. Box plots show mean,
        standard deviation, min-max and outliers of the difference between
        default and tuned models.
     Parameters
     ----------
     target_score : string
         The score to base the box plots on.
     results_path : string
         The file from which the tuned results can be retrieved.
     default_results_path : string
         The file from which the default results can be retrieved.
     """
    scores = cross_validation.get_evaluation_metric(target_score.lower(), results_path)
    default_scores = cross_validation.get_evaluation_metric(target_score.lower(),
                                                            default_results_path)

    diff = {key: [abs(scores[key][i] - default_scores[key][i]) for i in
                  range(len(scores[key]))] for key in scores.keys()}

    df = pd.DataFrame.from_dict(diff)
    _draw_boxplots(df, "{} improvement".format(target_score))


def per_fold_bar_chart_ensemble(models=ENSEMBLE_MODELS, target_score='accuracy',
                                results_path=utils.RESULTS_FILE_PATH,
                                results_ensemble_path=utils.ENSEMBLE_RESULTS_FILE_PATH):
    """ Plot comparative bar charts between ensemble performance and single
        models after 10-fold cross validation. Bars are computed per fold.
     Parameters
     ----------
     models : list
        List of models to be compared. Shouldn't include the ensemble itself.
     target_score : string
         The score to base the box plots on.
     results_path : string
         The file from which the tuned results can be retrieved.
     results_ensemble_path : string
         The file from which the ensemble results can be retrieved.
     """
    resuslts_single = cross_validation.get_evaluation_metric(target_score,
                                                             results_path)
    results_ensemble = cross_validation.get_evaluation_metric(target_score,
                                                              results_ensemble_path)
    colours = ["c", "purple", "darkorange", "pink"]
    models = [utils.get_model_name(m) for m in models]
    scores = {k: resuslts_single[k] for k in models}
    ensemble_name = list(results_ensemble.keys())[0]
    scores[ensemble_name] = results_ensemble[ensemble_name]
    _draw_bar_chart(scores, colours)


def performance_heatmap(target_score="accuracy",
                        results_path=utils.RESULTS_FILE_PATH):
    """ Draw a heat map to show the percentage of folds in which one model
        outperformed another. A model is said to outperform another when there
        is a difference of at least 0.01 in the target score.
     Parameters
     ----------
     target_score : string
         The score to base the box plots on.
     results_path : string
         The file from which the tuned results can be retrieved.
     """
    scores = cross_validation.get_evaluation_metric(target_score, results_path)
    upper_headers = list(scores.keys())
    side_headers = list(scores.keys())

    data = {uk: {sk: len(
        [i for i in range(len(scores[uk])) if
         scores[uk][i] > scores[sk][i] + 0.01])
        for sk in side_headers} for uk in upper_headers}

    clfs = sorted(upper_headers,
                  key=lambda x: sum([data[sk][x] for sk in side_headers]))
    print(clfs)
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
