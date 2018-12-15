import pandas as pd
from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

N_SPLITS = 3


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=296, kernel_initializer='uniform',
                    activation='softmax', kernel_constraint=maxnorm(4)))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu',
                    kernel_constraint=maxnorm(4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad',
                  metrics=['accuracy'])
    return model


data = pd.DataFrame()
for i in range(0, 9):
    split = pd.read_csv("../data/updrs_splits/split_{}.csv".format(i))
    data = pd.concat([data, split])

X = data.drop(columns=["patno", "score", "date_scan"])
y = data["score"].astype(int)

scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))

KerasNN = KerasClassifier(build_fn=baseline_model, epochs=2000,
                          batch_size=80, verbose=0, class_weight='balanced')
scores = cross_val_score(KerasNN, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(KerasNN, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_micro')
print("F1-micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(KerasNN, X, y,
                         cv=StratifiedShuffleSplit(n_splits=N_SPLITS),
                         scoring='f1_macro')
print("F1-macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
