import pandas as pd
from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold

N_SPLITS = 9


def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=296, kernel_initializer='uniform',
                    activation='softmax', kernel_constraint=maxnorm(4)))
    model.add(Dense(neurons, kernel_initializer='uniform', activation='relu',
                    kernel_constraint=maxnorm(4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad',
                  metrics=['accuracy'])
    return model


def process_data(df):
    X = df.drop(columns=["patno", "score", "date_scan"])
    y = df["score"].astype(int)
    scaler = preprocessing.StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))

    return X, y


data_test = pd.read_csv("../../../data/updrs_splits/split_9.csv")
data_train = pd.DataFrame()
for j in range(0, 9):
    split = pd.read_csv("../../../data/updrs_splits/split_{}.csv".format(j))
    data_train = pd.concat([data_train, split])
X_train, y_train = process_data(data_train)
X_test, y_test = process_data(data_test)

model = KerasClassifier(build_fn=create_model, verbose=0, batch_size=80,
                        epochs=2000)
neurons = [8, 16, 64, 128, 256, 512]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    cv=StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=0))
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
