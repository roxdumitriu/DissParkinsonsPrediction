import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}

data = pd.read_csv("../data/updrs.csv")
X = data.drop(columns=["patno", "score", "date_scan"])
y = data["score"].astype(int)
scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

score = "accuracy"

clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                   scoring='%s_macro' % score)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()