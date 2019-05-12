from sklearn.linear_model import LogisticRegression


def get_model():
    """ Get the Sklearn Logistic Regression classifier with tuned parameters.
    Returns
    ----------
    A Logistic Regression classifier with tuned parameters.
    """
    return LogisticRegression(C=1.5, max_iter=2000, fit_intercept=True,
                              multi_class='ovr', penalty='l1',
                              solver='liblinear')


def get_default_model():
    """ Get the default Sklearn Logistic Regression classifier. Used to
    see the improvement of hyperparameter tuning.
    Returns
    ----------
    The default Sklearn Logistic Regression classifier.
    """
    return LogisticRegression()
