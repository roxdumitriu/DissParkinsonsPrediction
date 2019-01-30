from sklearn.ensemble import RandomForestClassifier


def get_model():
    return RandomForestClassifier(n_estimators=500, max_features=0.25, criterion='entropy', oob_score=True)
