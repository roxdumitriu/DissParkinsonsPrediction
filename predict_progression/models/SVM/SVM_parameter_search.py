import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


def process_data(df):
    X = df.drop(columns=["patno", "score", "date_scan"])
    y = df["score"].astype(int)
    scaler = preprocessing.StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))

    return X, y


train_df = pd.DataFrame()
for x in range(0, 9):
    split = pd.read_csv("../data/updrs_splits/split_{}.csv".format(x))
    train_df = pd.concat([train_df, split])

test_df = pd.read_csv("../data/updrs_splits/split_9.csv")

X_train, y_train = process_data(train_df)
X_test, y_test = process_data(test_df)

max_acc = 0
max_c = 0
max_gamma = 0
max_y_pred = []
for gamma in [x / 10 for x in range(1, 10)]:
    for c in range(1, 1000, 50):
        clf = OneVsOneClassifier(svm.SVC(C=c, kernel='rbf', gamma=gamma))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_pred=y_pred, y_true=y_test)
        if acc > max_acc:
            max_acc = acc
            max_c = c
            max_gamma = gamma
            max_y_pred = y_pred

print("max accuracy: {0}\n C: {1}, gamma:{2}".format(max_acc, max_c, max_gamma))
print(confusion_matrix(y_test, max_y_pred))

