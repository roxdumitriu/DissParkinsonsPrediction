from sklearn.linear_model import SGDClassifier


def get_model():
    """ Get the Sklearn Stochastic Gradient Descent classifier with tuned
    parameters.
    Returns
    ----------
    A Stochastic Gradient Descent classifier with tuned parameters.
    """
    return SGDClassifier(tol=0.0001, loss="log",
                         penalty="l1", alpha=0.01, fit_intercept=False,
                         eta0=0.01, learning_rate="optimal")


def get_default_model():
    """ Get the default Sklearn Stochastic Gradient Descent classifier. Used to
    see the improvement of hyperparameter tuning.
    Returns
    ----------
    The default Sklearn Random Forest classifier.
    """
    return SGDClassifier()
