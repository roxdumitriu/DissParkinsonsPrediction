from sklearn.linear_model import SGDClassifier


def get_model():
    return SGDClassifier(tol=0.0001, loss="log",
                         penalty="l1", alpha=0.01, fit_intercept=False,
                         eta0=0.01, learning_rate="optimal")


def get_default_model():
    return SGDClassifier()
