from sklearn.ensemble import GradientBoostingClassifier


def get_model():
    return GradientBoostingClassifier(n_estimators=2500, max_depth=4,
                                      learning_rate=0.1,
                                      max_features="log2")
