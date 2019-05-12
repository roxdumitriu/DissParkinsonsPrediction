import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split


def get_data():
    """ Get the brain scans data and diagnosis. Encode the diagnosis into
        integer classes. Scale the data using standard scaling.
     Returns
     ----------
     A train-test split. 80% for training and 20% for testing.
     """
    data = pd.read_csv("../../data/thickness_and_volume_data.csv")
    X = data.drop(columns=["patno", "date_scan", "diagnosis"])
    y = data["diagnosis"]
    # 0 = Healthy, 1 = PD
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    return X_train, X_test, y_train, y_test


def get_model():
    """ Get the model to train the diagnosis prediction on.
     Returns
     ----------
     An Sklearn estimator.
     """
    return svm.SVC()


def evaluate_model():
    """ Evaluate the prediction on the brain scans data. Prints the accuracy
        score.
     """
    model = get_model()
    X_train, X_test, y_train, y_test = get_data()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Accuracy test set: %0.4f" % score)
