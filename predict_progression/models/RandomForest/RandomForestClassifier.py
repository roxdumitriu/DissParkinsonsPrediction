from sklearn.ensemble import RandomForestClassifier


def get_model():
    return RandomForestClassifier(n_estimators=1000, max_features='sqrt', criterion='gini')


def get_default_model():
    return RandomForestClassifier()
