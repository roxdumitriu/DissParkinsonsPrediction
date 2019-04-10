import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split

data = pd.read_csv("../../data//thickness_and_volume_data.csv")
le = preprocessing.LabelEncoder()

X = data.drop(columns=["patno", "date_scan", "diagnosis"])
cols = X.columns.values
y = data["diagnosis"]
le.fit(y)
# 0 = Healthy, 1 = PD
y = le.transform(y)
scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X), columns=cols)

SVClassifier = svm.SVC()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
SVClassifier.fit(X_train, y_train)
score = SVClassifier.score(X_test, y_test)
print("Accuracy test set: %0.4f" % score)
score = SVClassifier.score(X_train, y_train)
print("Accuracy training set: %0.4f" % score)

