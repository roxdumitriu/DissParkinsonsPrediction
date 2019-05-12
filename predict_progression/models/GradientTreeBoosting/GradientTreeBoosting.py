from sklearn.ensemble import GradientBoostingClassifier


def get_model():
    """ Get the Sklearn Gradient Tree Boosting classifier with tuned parameters.
    Returns
    ----------
    A Gradient Tree Boosting classifier with tuned parameters.
    """
    return GradientBoostingClassifier(n_estimators=2000,
                                      learning_rate=0.03,
                                      max_features="sqrt", subsample=0.6)


def get_default_model():
    """ Get the default Sklearn Gradient Tree Boosting classifier. Used to
    see the improvement of hyperparameter tuning.
    Returns
    ----------
    The default Sklearn Gradient Tree Boosting classifier.
    """
    return GradientBoostingClassifier()
