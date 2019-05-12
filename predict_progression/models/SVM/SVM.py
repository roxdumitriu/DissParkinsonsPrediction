from sklearn import svm


def get_model():
    """ Get the Sklearn Support Vector Machine classifier with tuned
    parameters.
    Returns
    ----------
    A Support Vector Machine classifier with tuned parameters.
    """
    return svm.SVC(C=0.01, gamma=0.1, kernel='poly', degree=3, coef0=10.0,
                   probability=True)


def get_default_model():
    """ Get the Sklearn Support Vector Machine classifier with default
    parameters. Used to see the improvement of hyperparameter tuning.
    Returns
    ----------
    A Support Vector Machine classifier with default parameters.
    """
    return svm.SVC()
