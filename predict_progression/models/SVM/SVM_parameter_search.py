from sklearn.svm import SVC

from predict_progression.models.parameter_search import parameter_search

tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['poly'], 'degree': [2, 3, 5],
     'coef0': [1.0, 10.0]},
]

parameter_search(model=SVC(class_weight='balanced'),
                 parameters=tuned_parameters, n_splits=10)
