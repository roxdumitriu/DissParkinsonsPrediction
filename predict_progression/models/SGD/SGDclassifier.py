from sklearn.linear_model import SGDClassifier


def get_model():
    return SGDClassifier(max_iter=3000, loss="squared_hinge",
                         penalty="none", alpha=0.01, fit_intercept=False,
                         eta0=0.001, learning_rate="invscaling")
