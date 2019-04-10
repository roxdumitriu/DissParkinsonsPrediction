from sklearn.ensemble import RandomForestClassifier

from predict_progression.models.parameter_search import parameter_search

parameters = [{
               'n_estimators': [1, 10, 100, 1000, 2000, 5000],
               'max_features': ['sqrt', 'log2', None],
               'criterion': ['gini', 'entropy']}
              ]
parameter_search(model=RandomForestClassifier(), parameters=parameters, n_splits=10)