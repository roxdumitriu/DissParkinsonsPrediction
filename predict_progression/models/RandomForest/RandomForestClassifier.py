from sklearn.ensemble import RandomForestClassifier


def get_model():
    """ Get the Sklearn Random Forest classifier with tuned parameters.
    Returns
    ----------
    A Random Forest classifier with tuned parameters.
    """
    return RandomForestClassifier(n_estimators=1000, max_features='sqrt', criterion='gini')


def get_default_model():
    """ Get the default Sklearn Random Forest classifier. Used to
    see the improvement of hyperparameter tuning.
    Returns
    ----------
    The default Sklearn Random Forest classifier.
    """
    return RandomForestClassifier()
