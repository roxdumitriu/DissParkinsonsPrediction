from sklearn.ensemble import RandomForestClassifier


def get_model():
    return RandomForestClassifier(n_estimators=1000, min_weight_fraction_leaf=0)
