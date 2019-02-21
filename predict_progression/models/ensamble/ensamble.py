from sklearn.ensemble import VotingClassifier

import \
    predict_progression.models.GradientTreeBoosting.GradientBoostingClassifier as gtb
import predict_progression.models.KerasNN.KerasNN as keras
import predict_progression.models.MaxEnt.MaxEntClassifier as maxent
import predict_progression.models.RandomForest.RandomForestClassifier as rfc
import predict_progression.models.SGD.SGDclassifier as sgd
import predict_progression.models.SVM.SVM as svm

MODELS = [('gtb', gtb.get_model()), ('svm', svm.get_model()),
          ('rfc', rfc.get_model())]


def get_model():
    return VotingClassifier(estimators=MODELS, voting='hard')
