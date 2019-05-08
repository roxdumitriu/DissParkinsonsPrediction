from sklearn.ensemble import VotingClassifier

import \
    predict_progression.models.GradientTreeBoosting.GradientTreeBoosting as gtb
import predict_progression.models.KerasNN.KerasNN as keras
import predict_progression.models.SGD.SGDclassifier as sgd

MODELS = [('gtb', gtb.get_model()), ('mlp', keras.get_model()),
          ('sgd', sgd.get_model())]


def get_model():
    return VotingClassifier(estimators=MODELS, voting='soft', weights=[5, 2, 5])
