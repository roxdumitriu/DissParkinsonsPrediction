from sklearn.ensemble import GradientBoostingClassifier

from predict_progression.models.parameter_search import *

parameters = {'n_estimators': [1, 10, 100, 1000, 2000, 5000],
               'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09],
               'max_features': [None, "sqrt", "log2"],
               'subsample': [0.5, 0.6, 0.7, 0.8, 1]}
parameter_search(model=GradientBoostingClassifier(), parameters=parameters,
                 n_splits=10)


