from sklearn.ensemble import GradientBoostingClassifier

from predict_progression.models.parameter_search import *

parameters = [{'n_estimators': [10, 50, 100, 1000, 2000, 3000],
               'learning_rate': [0.1, 0.01, 0.001],
               'max_depth': [2, 4, 8, 16],
               'max_features': [None, "auto", "log2"]}
              ]
parameter_search(model=GradientBoostingClassifier(), parameters=parameters,
                 n_splits=10)


