from sklearn.ensemble import GradientBoostingClassifier


def get_model():
    return GradientBoostingClassifier(n_estimators=2500, max_depth=4,
                                      learning_rate=0.01,
                                      max_features="log2", subsample=0.5, validation_fraction=0.2, n_iter_no_change=5, tol=0.01,)
