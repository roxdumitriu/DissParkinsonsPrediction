import numpy as np

import \
    predict_progression.models.GradientTreeBoosting.GradientBoostingClassifier as gtb
import predict_progression.models.KerasNN.KerasNN as keras
import predict_progression.models.MaxEnt.MaxEntClassifier as maxent
import predict_progression.models.RandomForest.RandomForestClassifier as rfc
import predict_progression.models.SGD.SGDclassifier as sgd
import predict_progression.models.SVM.SVM as svm
from predict_progression.evaluation import utils

MODELS = [gtb.get_model(), keras.get_model(),
          maxent.get_model(), sgd.get_model(), rfc.get_model(), svm.get_model()]

SPLITS = ["../../data/updrs_splits/split_{}.csv".format(i) for i in range(0, 8)]


def get_predictions(model, splits):
    predictions = np.array([])
    for i in range(0, len(splits)):
        X_train, y_train, X_test, y_test = utils.train_and_test_from_splits(
            splits, i)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions = np.concatenate([predictions, pred])
    np.savetxt("predictions/{}_all_predictions.txt".format(
        utils.get_model_name(model)), predictions)


for model in MODELS:
    get_predictions(model, SPLITS)
    print(utils.get_model_name(model), "done")
