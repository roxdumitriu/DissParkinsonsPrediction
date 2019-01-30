from sklearn.ensemble import RandomForestClassifier

from predict_progression.models.parameter_search import parameter_search

parameters = [{
               'n_estimators': [1, 10, 100, 500, 1000],
               'min_weight_fraction_leaf': [0, 0.1, 0.25, 0.5],
               'max_features': ['sqrt', 'log2', None, 0.25, 0.5, 5],
               'criterion': ['gini', 'entropy']}
              ]
parameter_search(model=RandomForestClassifier(n_estimators=1000, min_weight_fraction_leaf=0, ), parameters=parameters, n_splits=10)