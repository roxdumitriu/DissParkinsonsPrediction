import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


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

tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [1 / (10 ** x) for x in range(3, 8)],
     'C': [1, 3, 10, 50, 100, 500, 1000, 1500]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(class_weight="balanced"), tuned_parameters,
                       cv=StratifiedKFold(n_splits=5, shuffle=True,
                                          random_state=0),
                       scoring='%s' % score)
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
    print(confusion_matrix(y_true, y_pred))
    print()
