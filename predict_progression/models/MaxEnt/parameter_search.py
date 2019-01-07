from sklearn.linear_model import LogisticRegression

from predict_progression.models.parameter_search import parameter_search

parameters = [{'penalty': ['l1'],
               'solver': ['liblinear', 'saga']},
              {'penalty': ['l2'],
               'solver': ['newton-cg', 'lbfgs', 'sag']}
              ]

parameter_search(
    model=LogisticRegression(C=1.5, max_iter=2000, fit_intercept=True,
                             multi_class='auto'), parameters=parameters,
    n_splits=10)
