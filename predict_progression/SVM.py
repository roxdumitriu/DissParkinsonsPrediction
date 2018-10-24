import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

N_SPLITS = 1
data = pd.read_csv("../data/updrs.csv")
le = preprocessing.LabelEncoder()

X = data.drop(columns=["patno", "score", "date_scan"])
y = data["score"].astype(int)

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

average_accuracy = sum(fscores)/N_SPLITS
print('Average fscore: {0:0.2f}'.format(average_accuracy))
