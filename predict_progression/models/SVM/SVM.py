from sklearn import svm


def get_model():
    return svm.SVC(C=0.01, gamma=0.1, kernel='poly', degree=3, coef0=10.0)
