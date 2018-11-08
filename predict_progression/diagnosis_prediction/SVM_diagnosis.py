import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

N_SPLITS = 10
data = pd.read_csv("../../data/diagnosis.csv")
le = preprocessing.LabelEncoder()

X = data.drop(columns=["patno", "date_scan", "diagnosis"])
y = data["diagnosis"]
le.fit(y)
# 0 = Healthy, 1 = PD, 2 = PRODROMA
y = le.transform(y)
scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))

skf = StratifiedShuffleSplit(n_splits=N_SPLITS)
fscores_macro = []
accuracies = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svclassifier = svm.SVC(kernel='linear', gamma='auto')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    fscores_macro.append(fscore_macro)
    accuracies.append(accuracy_score(y_test, y_pred))

average_f1_macro = sum(fscores_macro) / N_SPLITS
average_accuracy = sum(accuracies) / N_SPLITS
print('Average fscore: {0:0.2f}'.format(average_f1_macro))
print('Average accuracy: {0:0.2f}'.format(average_accuracy))

SVClassifier = svm.SVC(kernel='linear', gamma='auto')
scores = cross_val_score(SVClassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_micro')
print("F1-micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(SVClassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_macro')
print("F1-macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(SVClassifier, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
