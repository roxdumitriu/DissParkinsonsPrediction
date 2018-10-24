import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

N_SPLITS = 3
data = pd.read_csv("../data/diagnosis.csv")
le = preprocessing.LabelEncoder()

X = data.drop(columns=["patno", "date_scan", "diagnosis"])
y = data["diagnosis"]
le.fit(y)
# 0 = Healthy, 1 = PD, 2 = PRODROMA
y = le.transform(y)

skf = StratifiedShuffleSplit(n_splits=N_SPLITS)
fscores = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svclassifier = svm.SVC(kernel='linear', gamma='auto')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    fscore = f1_score(y_test, y_pred, average='macro')
    fscores.append(fscore)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

average_accuracy = sum(fscores)/N_SPLITS
print('Average fscore: {0:0.2f}'.format(average_accuracy))
