from sklearn.ensemble import GradientBoostingClassifier


def get_model():
    return GradientBoostingClassifier(n_estimators=2000,
                                      learning_rate=0.03,
                                      max_features="sqrt", subsample=0.6)


def get_default_model():
    return GradientBoostingClassifier()
