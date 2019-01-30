import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from predict_progression.evaluation.utils import *

MODELS = ALL_MODELS
MODELS_PREDCITIONS = {
get_model_name(model): "predictions/{}_all_predictions.txt".format(
    get_model_name(model)) for model in MODELS}


def get_predictions(models, n_splits=10):
    X, y = get_data()
    np.savetxt("predictions/all_predictions.txt", X=y)
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for model in models:
        model_name = get_model_name(model)
        if "Keras" in model_name:
            es = EarlyStopping(monitor="acc")
            s = cross_val_predict(model, X, y, cv=sss,
                                  fit_params={'callbacks': [es]})
        else:
            s = cross_val_predict(model, X, y, cv=sss)
        print(s.shape)
        np.savetxt("predictions/{}_all_predictions.txt".format(model_name), X=s)


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
    fig, ax = plt.subplots()
    im = ax.imshow(preds, cmap="YlGn")
    ax.set_xticks(np.arange(len(headers)))
    ax.set_yticks(np.arange(len(headers)))
    ax.set_xticklabels(headers)
    ax.set_yticklabels(headers)

    for i in range(len(headers)):
        for j in range(len(headers)):
            text = ax.text(j, i, preds[i][j], ha="center", va="center",
                           color="black" if 0.5 > preds[i][j] > 0 else "white")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    # fig.tight_layout()
    plt.show()


# print(wilcoxon_test("predictions/GradientBoostingClassifier_all_predictions.txt", "predictions/KerasNeuralNet_all_predictions.txt"))
significance_heatmap(MODELS_PREDCITIONS)
