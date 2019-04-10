from sklearn.linear_model import LogisticRegression


def get_model():
    return LogisticRegression(C=1.5, max_iter=2000, fit_intercept=True,
                              multi_class='ovr', penalty='l1',
                              solver='liblinear')


def get_default_model():
    return LogisticRegression()
