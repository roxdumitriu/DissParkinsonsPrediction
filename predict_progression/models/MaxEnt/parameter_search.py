from sklearn.linear_model import LogisticRegression

from predict_progression.models.parameter_search import parameter_search

parameters = [{'penalty': ['l1', 'l2'],
               'C': [0.1, 1.5, 10, 100, 100, 1000]}
              ]

parameter_search(model=LogisticRegression(max_iter=2000, fit_intercept=True,
                                          multi_class='ovr',
                                          solver='liblinear'),
                 parameters=parameters, n_splits=10)
