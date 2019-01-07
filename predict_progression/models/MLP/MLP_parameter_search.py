import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

from predict_progression.models.parameter_search import parameter_search

parameters = [{'alpha': [0.1, 0.01, 0.001],
               'hidden_layer_sizes': [(256, 64, 64)],
               'activation': ['identity', 'logistic', 'tanh', 'relu'],
               'learning_rate': ['constant', 'invscaling', 'adaptive'],
               'max_iter': [5000, 10000]},
              ]
parameter_search(model=MLPClassifier(), parameters=parameters, n_splits=10)