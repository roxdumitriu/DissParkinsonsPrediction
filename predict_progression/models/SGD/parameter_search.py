from sklearn.linear_model import SGDClassifier

from predict_progression.models.parameter_search import parameter_search

parameters = [
    {'learning_rate': ["invscaling"],
     'eta0': [100, 10, 1, 0.1, 0.01],
     'power_t': [-1, 1, 2, 3],
     'alpha': [0.1, 0.01, 0.001, 0.0001],
     'fit_intercept': [True, False],
     },
    {'learning_rate': ["constant"],
     'eta0': [100, 10, 1, 0.1, 0.01],
     'alpha': [0.1, 0.01, 0.001, 0.0001],
     'fit_intercept': [True, False],
     },
    {'learning_rate': ["optimal"],
     'eta0': [100, 10, 1, 0.1, 0.01],
     'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
     'fit_intercept': [True, False],
     'tol':[0.1, 0.01, 0.001, 0.0001]
     }
]

parameter_search(
    model=SGDClassifier(loss='log', penalty='l1', shuffle=True, early_stopping=True),
    parameters=parameters, n_splits=10)
