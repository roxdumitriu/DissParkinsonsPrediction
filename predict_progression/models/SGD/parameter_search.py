from sklearn.linear_model import SGDClassifier

from predict_progression.models.parameter_search import parameter_search

parameters = [{
    'loss': ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
    'penalty': ["none", "l2", "l1", "elasticnet"],
    'learning_rate': ["constant", "optimal", "invscaling", "adaptive"],
    'eta0': [0.001, 0.0001],
    'alpha': [0.1, 0.01, 0.001, 0.0001, 0.000001],
    'fit_intercept': [True, False],
    'l1_ratio': [x / 100 for x in range(10, 30, 5)]
}
]

parameter_search(model=SGDClassifier(), parameters=parameters, n_splits=10)
